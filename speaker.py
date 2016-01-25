import colorsys
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import crossentropy_categorical_1hot
from lasagne.layers import InputLayer, DropoutLayer, EmbeddingLayer, NonlinearityLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output
from lasagne.layers.recurrent import LSTMLayer, Gate
from lasagne.init import Constant
from lasagne.nonlinearities import softmax
from lasagne.updates import rmsprop

from bt import config, progress, iterators
from bt.rng import get_rng
from neural import NeuralLearner, SimpleLasagneModel

parser = config.get_options_parser()
parser.add_argument('--speaker_cell_size', type=int, default=20)
parser.add_argument('--speaker_forget_bias', type=float, default=5.0)
parser.add_argument('--speaker_color_resolution', type=int, default=4)
parser.add_argument('--speaker_eval_batch_size', type=int, default=16384)

rng = get_rng()


class UniformPrior(object):
    def fit(self, xs, ys):
        pass

    def apply(self, input_vars):
        c, _, _ = input_vars
        return -3.0 * np.log(256.0) * T.ones_like(c[:, 0])


class SpeakerLearner(NeuralLearner):
    '''
    An speaker with a feedforward neural net color input passed into an LSTM
    to generate a description.
    '''
    def __init__(self, id=None):
        options = config.options()
        super(SpeakerLearner, self).__init__(options.speaker_color_resolution, id=id)

    def predict(self, eval_instances, random=False):
        options = config.options()

        result = []
        batches = iterators.iter_batches(eval_instances, options.speaker_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // options.speaker_eval_batch_size + 1

        print('Predicting')
        progress.start_task('Predict batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)

            (c, _p, mask), (_y,) = self._data_to_arrays(batch, test=True)

            done = np.zeros((len(batch),), dtype=np.bool)
            outputs = [['<s>'] + ['<MASK>'] * (self.seq_vec.max_len - 2)
                       for _ in batch]
            length = 0
            while not done.all() and length < self.seq_vec.max_len - 1:
                p = self.seq_vec.vectorize_all(outputs)
                preds = self.model.predict([c, p, mask])
                if random:
                    indices = sample(preds[:, length, :])
                else:
                    indices = preds[:, length, :].argmax(axis=1)
                for out, idx in zip(outputs, indices):
                    token = self.seq_vec.indices_token[idx]
                    if length + 1 < self.seq_vec.max_len - 1:
                        out[length + 1] = token
                    else:
                        out.append(token)
                done = np.logical_or(done, indices == self.seq_vec.token_indices['</s>'])
                length += 1
            result.extend([' '.join(strip_invalid_tokens(o)) for o in outputs])
        progress.end_task()

        return result

    def score(self, eval_instances):
        options = config.options()

        result = []
        batches = iterators.iter_batches(eval_instances, options.speaker_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // options.speaker_eval_batch_size + 1

        print('Scoring')
        progress.start_task('Score batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)

            xs, (n,) = self._data_to_arrays(batch, test=True)
            _, _, mask = xs

            probs = self.model.predict(xs)
            token_probs = probs[np.arange(probs.shape[0])[:, np.newaxis],
                                np.arange(probs.shape[1]), n]
            scores_arr = np.sum(np.log(token_probs) * mask, axis=1)
            scores = scores_arr.tolist()
            result.extend(scores)
        progress.end_task()

        return result

    def sample(self, inputs):
        return self.predict(inputs, random=True)

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_color, get_desc = (get_o, get_i) if inverted else (get_i, get_o)

        if init_vectorizer:
            self.seq_vec.add_all(['<s>'] + get_desc(inst).split() + ['</s>']
                                 for inst in training_instances)

        colors = []
        previous = []
        next_tokens = []
        for i, inst in enumerate(training_instances):
            desc, (hue, sat, val) = get_desc(inst), get_color(inst)
            color_0_1 = colorsys.hsv_to_rgb(hue / 360.0, sat / 100.0, val / 100.0)
            color = tuple(min(d * 256, 255) for d in color_0_1)
            if test:
                full = ['<s>'] + ['<MASK>'] * (self.seq_vec.max_len - 2)
            else:
                desc = desc.split()
                full = (['<s>'] + desc + ['</s>'] +
                        ['<MASK>'] * (self.seq_vec.max_len - 2 - len(desc)))
            prev = full[:-1]
            next = full[1:]
            # print('%s, %s -> %s' % (repr(color), repr(prev), repr(next)))
            colors.append(color)
            previous.append(prev)
            next_tokens.append(next)

        c = np.zeros((len(colors),), dtype=np.int32)
        P = np.zeros((len(previous), self.seq_vec.max_len - 1), dtype=np.int32)
        mask = np.zeros((len(previous), self.seq_vec.max_len - 1), dtype=np.int32)
        N = np.zeros((len(next_tokens), self.seq_vec.max_len - 1), dtype=np.int32)
        c[:] = self.color_vec.vectorize_all(colors)
        for i, (color, prev, next) in enumerate(zip(colors, previous, next_tokens)):
            if len(prev) > P.shape[1]:
                prev = prev[:P.shape[1]]
            if len(next) > N.shape[1]:
                next = next[:N.shape[1]]
            P[i, :len(prev)] = self.seq_vec.vectorize(prev)
            N[i, :len(next)] = self.seq_vec.vectorize(next)
            for t, token in enumerate(next):
                mask[i, t] = (token != '<MASK>')
        c = np.tile(c[:, np.newaxis], [1, self.seq_vec.max_len - 1])

        return [c, P, mask], [N]

    def _build_model(self, model_class=SimpleLasagneModel):
        id_tag = (self.id + '/') if self.id else ''
        input_vars = [T.imatrix(id_tag + 'inputs'),
                      T.imatrix(id_tag + 'previous'),
                      T.imatrix(id_tag + 'mask')]
        target_var = T.imatrix(id_tag + 'targets')

        self.l_out, self.input_layers = self. _get_l_out(input_vars)
        self.model = model_class(input_vars, [target_var], self.l_out, id=self.id,
                                 loss=self.masked_loss(input_vars), optimizer=rmsprop)

        self.prior_emp = UniformPrior()
        self.prior_smooth = UniformPrior()

    def _get_l_out(self, input_vars):
        options = config.options()
        id_tag = (self.id + '/') if self.id else ''

        input_var, prev_output_var, mask_var = input_vars

        l_color = InputLayer(shape=(None, self.seq_vec.max_len - 1), input_var=input_var,
                             name=id_tag + 'color_input')
        l_color_embed = EmbeddingLayer(l_color, input_size=self.color_vec.num_types,
                                       output_size=options.speaker_cell_size,
                                       name=id_tag + 'color_embed')
        l_prev_out = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                                input_var=prev_output_var,
                                name=id_tag + 'prev_input')
        l_prev_embed = EmbeddingLayer(l_prev_out, input_size=len(self.seq_vec.tokens),
                                      output_size=options.speaker_cell_size,
                                      name=id_tag + 'prev_embed')
        l_in = ConcatLayer([l_color_embed, l_prev_embed], axis=2, name=id_tag + 'color_prev')
        l_mask_in = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                               input_var=mask_var, name=id_tag + 'mask_input')
        l_lstm1 = LSTMLayer(l_in, mask_input=l_mask_in, num_units=options.speaker_cell_size,
                            forgetgate=Gate(b=Constant(options.speaker_forget_bias)),
                            name=id_tag + 'lstm1')
        l_lstm1_drop = DropoutLayer(l_lstm1, p=0.2, name=id_tag + 'lstm1_drop')
        l_lstm2 = LSTMLayer(l_lstm1_drop, num_units=len(self.seq_vec.tokens),
                            forgetgate=Gate(b=Constant(options.speaker_forget_bias)),
                            name=id_tag + 'lstm2')
        l_shape = ReshapeLayer(l_lstm2, (-1, len(self.seq_vec.tokens)),
                               name=id_tag + 'reshape')
        l_softmax = NonlinearityLayer(l_shape, nonlinearity=softmax,
                                      name=id_tag + 'softmax')
        l_out = ReshapeLayer(l_softmax, (-1, self.seq_vec.max_len - 1, len(self.seq_vec.tokens)),
                             name=id_tag + 'out')

        return l_out, [l_color, l_prev_out, l_mask_in]

    def loss_out(self, input_vars=None, target_var=None):
        if input_vars is None:
            input_vars = self.model.input_vars
        if target_var is None:
            target_var = self.model.target_var
        pred = get_output(self.l_out, dict(zip(self.input_layers, input_vars)))
        loss = self.masked_loss(input_vars)
        return loss(pred, target_var)

    def masked_loss(self, input_vars):
        return masked_seq_crossentropy(input_vars[-1])


def sample(a, temperature=1.0):
    a = np.array(a)
    if len(a.shape) < 1:
        raise ValueError('scalar is not a valid probability distribution')
    elif len(a.shape) == 1:
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(rng.multinomial(1, a, 1))
    else:
        return np.array([sample(s, temperature) for s in a])


def strip_invalid_tokens(sentence):
    good_tokens = [t for t in sentence if t not in ('<s>', '<MASK>')]
    if '</s>' in good_tokens:
        end_pos = good_tokens.index('</s>')
        good_tokens = good_tokens[:end_pos]
    return good_tokens


def crossentropy_categorical_1hot_nd(coding_dist, true_idx):
    '''
    A n-dimensional generalization of `theano.tensor.nnet.crossentropy_categorical`.

    :param coding_dist: a float tensor with the last dimension equal to the number of categories
    :param true_idx: an integer tensor with one fewer dimension than `coding_dist`, giving the
                     indices of the true targets
    '''
    if coding_dist.ndim != true_idx.ndim + 1:
        raise ValueError('`coding_dist` must have one more dimension that `true_idx` '
                         '(got %s and %s)' % (coding_dist.type, true_idx.type))
    coding_flattened = T.reshape(coding_dist, (-1, T.shape(coding_dist)[-1]))
    scores_flattened = crossentropy_categorical_1hot(coding_flattened, true_idx.flatten())
    return T.reshape(scores_flattened, true_idx.shape)


def masked_seq_crossentropy(mask):
    '''
    Return a loss function for sequence models.

    :param mask: a 2-D int tensor (num_examples x max_length) with 1 in valid token locations
        and 0 in locations that should be masked out

    The returned function will have the following parameters and return type:

    :param coding_dist: a 3-D float tensor (num_examples x max_length x num_token_types)
        of log probabilities assigned to each token
    :param true_idx: a 2-D int tensor (num_examples x max_length) of true token indices
    :return: a 1-D float tensor of per-example cross-entropy values
    '''
    def msxe_loss(coding_dist, true_idx):
        mask_float = T.cast(mask, 'float32')
        return (crossentropy_categorical_1hot_nd(coding_dist, true_idx) * mask_float).sum(axis=1)

    return msxe_loss
