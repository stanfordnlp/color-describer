import colorsys
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import crossentropy_categorical_1hot
from lasagne.layers import InputLayer, DropoutLayer, EmbeddingLayer, NonlinearityLayer
from lasagne.layers import ConcatLayer, ReshapeLayer
from lasagne.layers.recurrent import LSTMLayer, Gate
from lasagne.init import Constant
from lasagne.nonlinearities import softmax
from lasagne.updates import rmsprop

from bt import config
from bt.rng import get_rng
from neural import NeuralLearner, SimpleLasagneModel

parser = config.get_options_parser()
parser.add_argument('--speaker_cell_size', type=int, default=20)
parser.add_argument('--speaker_forget_bias', type=float, default=5.0)
parser.add_argument('--speaker_color_resolution', type=int, default=4)

rng = get_rng()


class SpeakerLearner(NeuralLearner):
    '''
    An speaker with a feedforward neural net color input passed into an LSTM
    to generate a description.
    '''
    def __init__(self):
        options = config.options()
        super(SpeakerLearner, self).__init__(options.speaker_color_resolution)

    def predict(self, eval_instances, random=False):
        (c, _p, mask), y = self._data_to_arrays(eval_instances, test=True)

        print('Testing')
        done = np.zeros((len(eval_instances),), dtype=np.bool)
        outputs = [['<s>'] + ['<MASK>'] * (self.seq_vec.max_len - 2)
                   for _ in eval_instances]
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
        return [strip_invalid_tokens(o) for o in outputs]

    def score(self, eval_instances):
        xs, (y,) = self._data_to_arrays(eval_instances, test=True)
        _, _, mask = xs

        print('Testing')
        probs = self.model.predict(xs)
        token_probs = probs[np.arange(probs.shape[0])[:, np.newaxis],
                            np.arange(probs.shape[1]), y]
        scores_arr = np.sum(-np.log(token_probs) * mask, axis=1)
        scores = scores_arr.tolist()
        return scores

    def log_prior_emp(self, input_vars):
        raise NotImplementedError

    def log_prior_smooth(self, input_vars):
        # TODO
        return self.log_prior_emp(input_vars)

    def sample(self, inputs):
        raise NotImplementedError

    def _data_to_arrays(self, training_instances, test=False):
        if not test:
            self.seq_vec.add_all(['<s>'] + inst.output.split() + ['</s>']
                                 for inst in training_instances)

        colors = []
        previous = []
        next_tokens = []
        for i, inst in enumerate(training_instances):
            desc, (hue, sat, val) = inst.output.split(), inst.input
            color_0_1 = colorsys.hsv_to_rgb(hue / 360.0, sat / 100.0, val / 100.0)
            color = tuple(min(d * 256, 255) for d in color_0_1)
            full = ['<s>'] + desc + ['</s>'] + ['<MASK>'] * (self.seq_vec.max_len - 2 - len(desc))
            prev = full[:-1]
            next = full[1:]
            # print('%s, %s -> %s' % (repr(color), repr(prev), repr(next)))
            colors.append(color)
            previous.append(prev)
            next_tokens.append(next)
        print('Number of sequences: %d' % len(colors))

        print('Vectorization...')
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
        input_vars = [T.imatrix('inputs'),
                      T.imatrix('previous'),
                      T.imatrix('mask')]
        target_var = T.imatrix('targets')

        l_out = self._get_l_out(input_vars)
        self.model = model_class(input_vars, [target_var], l_out,
                                 loss=crossentropy_categorical_1hot_nd, optimizer=rmsprop)

    def _get_l_out(self, input_vars):
        options = config.options()

        input_var, prev_output_var, mask_var = input_vars

        l_color = InputLayer(shape=(None, self.seq_vec.max_len - 1), input_var=input_var)
        l_color_embed = EmbeddingLayer(l_color, input_size=self.color_vec.num_types,
                                       output_size=options.speaker_cell_size)
        l_prev_out = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                                input_var=prev_output_var)
        l_prev_embed = EmbeddingLayer(l_prev_out, input_size=len(self.seq_vec.tokens),
                                      output_size=options.speaker_cell_size)
        l_in = ConcatLayer([l_color_embed, l_prev_embed], axis=2)
        l_mask_in = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                               input_var=mask_var)
        l_lstm1 = LSTMLayer(l_in, mask_input=l_mask_in, num_units=options.speaker_cell_size,
                            forgetgate=Gate(b=Constant(options.speaker_forget_bias)))
        l_lstm1_drop = DropoutLayer(l_lstm1, p=0.2)
        l_lstm2 = LSTMLayer(l_lstm1_drop, num_units=len(self.seq_vec.tokens),
                            forgetgate=Gate(b=Constant(options.speaker_forget_bias)))
        l_shape = ReshapeLayer(l_lstm2, (-1, len(self.seq_vec.tokens)))
        l_softmax = NonlinearityLayer(l_shape, nonlinearity=softmax)
        return ReshapeLayer(l_softmax, (-1, self.seq_vec.max_len - 1, len(self.seq_vec.tokens)))


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
    return crossentropy_categorical_1hot(T.reshape(coding_dist, (-1, T.shape(coding_dist)[-1])),
                                         true_idx.flatten())
