import colorsys
import numpy as np
import theano
import theano.tensor as T
from collections import Counter
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, EmbeddingLayer, NonlinearityLayer
from lasagne.layers.recurrent import LSTMLayer, Gate
from lasagne.init import Constant
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from lasagne.updates import rmsprop

from bt import config, instance, progress, iterators
from neural import NeuralLearner, SimpleLasagneModel

parser = config.get_options_parser()
parser.add_argument('--listener_cell_size', type=int, default=20)
parser.add_argument('--listener_forget_bias', type=float, default=5.0)
parser.add_argument('--listener_color_resolution', type=int, default=4)
parser.add_argument('--listener_eval_batch_size', type=int, default=65536)


class UnigramPrior(object):
    def __init__(self, vocab_size, mask_index=None):
        self.counts = np.zeros((vocab_size,), dtype=np.int32)
        self.total = 0
        self.mask_index = mask_index
        self.log_probs = None

    def fit(self, xs, ys):
        (x,) = xs
        counts = np.bincount(x.flatten(), minlength=self.counts.shape[0])
        if self.mask_index is not None:
            counts[self.mask_index] = 0
        self.counts += counts
        self.total += np.sum(counts)
        self.log_probs = None

    def apply(self, input_vars):
        (x,) = input_vars
        if self.log_probs is None:
            self.log_probs = theano.shared(np.log(self.counts / self.total).astype(np.float32))

        token_probs = self.log_probs[x]
        if self.mask_index is not None:
            token_probs = token_probs * T.cast((x == self.mask_index), 'float32')
        return token_probs.sum()


class ListenerLearner(NeuralLearner):
    '''
    An LSTM-based listener (guesses colors from descriptions).
    '''
    def __init__(self, id=None):
        options = config.options()
        self.word_counts = Counter()
        super(ListenerLearner, self).__init__(options.listener_color_resolution, id=id)

    def predict_and_score(self, eval_instances, random=False):
        options = config.options()

        predictions = []
        scores = []
        batches = iterators.iter_batches(eval_instances, options.listener_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // options.listener_eval_batch_size + 1

        print('Testing')
        progress.start_task('Eval batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)

            xs, (y,) = self._data_to_arrays(batch, test=True)

            probs = self.model.predict(xs)
            predictions.extend(self.color_vec.unvectorize_all(probs.argmax(axis=1),
                                                              random=random, hsv=True))
            bucket_volume = (256.0 ** 3) / self.color_vec.num_types
            scores_arr = np.log(probs[np.arange(len(batch)), y]) - np.log(bucket_volume)
            scores.extend(scores_arr.tolist())
        progress.end_task()

        return predictions, scores

    def on_iter_end(self, step, writer):
        most_common = [desc for desc, count in self.word_counts.most_common(10)]
        insts = [instance.Instance(input=desc) for desc in most_common]
        xs, (y,) = self._data_to_arrays(insts, test=True)
        probs = self.model.predict(xs)
        for i, desc in enumerate(most_common):
            dist = probs[i, :]
            for image, channel in zip(self.color_vec.visualize_distribution(dist), '012'):
                writer.log_image(step, 'listener/%s/%s' % (desc, channel), image)
        super(ListenerLearner, self).on_iter_end(step, writer)

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_desc, get_color = (get_o, get_i) if inverted else (get_i, get_o)

        if init_vectorizer:
            self.seq_vec.add_all(['<s>'] + get_desc(inst).split() + ['</s>']
                                 for inst in training_instances)

        sentences = []
        colors = []
        for i, inst in enumerate(training_instances):
            self.word_counts.update([get_desc(inst)])
            desc = get_desc(inst).split()
            if get_color(inst):
                (hue, sat, val) = get_color(inst)
            else:
                assert test
                hue = sat = val = 0.0
            color_0_1 = colorsys.hsv_to_rgb(hue / 360.0, sat / 100.0, val / 100.0)
            assert all(0.0 <= d <= 1.0 for d in color_0_1), (get_color(inst), color_0_1)
            color = tuple(min(d * 256, 255) for d in color_0_1)
            s = ['<s>'] * (self.seq_vec.max_len - 1 - len(desc)) + desc
            s.append('</s>')
            # print('%s -> %s' % (repr(s), repr(color)))
            sentences.append(s)
            colors.append(color)

        x = np.zeros((len(sentences), self.seq_vec.max_len), dtype=np.int32)
        y = np.zeros((len(sentences),), dtype=np.int32)
        for i, sentence in enumerate(sentences):
            x[i, :] = self.seq_vec.vectorize(sentence)
            y[i] = self.color_vec.vectorize(colors[i])

        return [x], [y]

    def sample(self, inputs):
        return self.predict_and_score(inputs, random=True)[0]

    def _build_model(self, model_class=SimpleLasagneModel):
        id_tag = (self.id + '/') if self.id else ''
        input_var = T.imatrix(id_tag + 'inputs')
        target_var = T.ivector(id_tag + 'targets')

        l_out = self._get_l_out([input_var])

        self.model = model_class([input_var], [target_var], l_out,
                                 loss=categorical_crossentropy, optimizer=rmsprop, id=self.id)

        self.prior_emp = UnigramPrior(vocab_size=len(self.seq_vec.tokens))
        self.prior_smooth = UnigramPrior(vocab_size=len(self.seq_vec.tokens))  # TODO: smoothing

    def _get_l_out(self, input_vars):
        options = config.options()
        id_tag = (self.id + '/') if self.id else ''

        input_var = input_vars[0]

        l_in = InputLayer(shape=(None, self.seq_vec.max_len), input_var=input_var,
                          name=id_tag + 'desc_input')
        l_in_embed = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                                    output_size=options.listener_cell_size,
                                    name=id_tag + 'desc_embed')
        l_lstm1 = LSTMLayer(l_in_embed, num_units=options.listener_cell_size,
                            forgetgate=Gate(b=Constant(options.listener_forget_bias)),
                            name=id_tag + 'lstm1')
        l_lstm1_drop = DropoutLayer(l_lstm1, p=0.2, name=id_tag + 'lstm1_drop')
        l_lstm2 = LSTMLayer(l_lstm1_drop, num_units=options.listener_cell_size,
                            forgetgate=Gate(b=Constant(options.listener_forget_bias)),
                            name=id_tag + 'lstm2')
        l_lstm2_drop = DropoutLayer(l_lstm2, p=0.2, name=id_tag + 'lstm2_drop')

        l_hidden = DenseLayer(l_lstm2_drop, num_units=options.listener_cell_size, nonlinearity=None,
                              name=id_tag + 'hidden')
        l_hidden_drop = DropoutLayer(l_hidden, p=0.2, name=id_tag + 'hidden_drop')
        l_scores = DenseLayer(l_hidden_drop, num_units=self.color_vec.num_types, nonlinearity=None,
                              name=id_tag + 'scores')
        return NonlinearityLayer(l_scores, nonlinearity=softmax, name=id_tag + 'out')
