import colorsys
import numpy as np
import theano.tensor as T
from collections import Counter
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, EmbeddingLayer, NonlinearityLayer
from lasagne.layers.recurrent import LSTMLayer, Gate
from lasagne.init import Constant
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from lasagne.updates import rmsprop

from bt import config, instance
from neural import NeuralLearner, SimpleLasagneModel

parser = config.get_options_parser()
parser.add_argument('--listener_cell_size', type=int, default=20)
parser.add_argument('--listener_forget_bias', type=float, default=5.0)
parser.add_argument('--listener_color_resolution', type=int, default=4)


class ListenerLearner(NeuralLearner):
    '''
    An LSTM-based listener (guesses colors from descriptions).
    '''
    def __init__(self):
        options = config.options()
        self.word_counts = Counter()
        super(ListenerLearner, self).__init__(options.listener_color_resolution)

    def predict_and_score(self, eval_instances):
        xs, y = self._data_to_arrays(eval_instances, test=True)

        print('Testing')
        probs = self.model.predict(xs)
        predict = self.color_vec.unvectorize_all(probs.argmax(axis=1))
        bucket_volume = (256.0 ** 3) / self.color_vec.num_types
        scores_arr = np.log(bucket_volume) - np.log(probs[np.arange(len(eval_instances)), y])
        scores = scores_arr.tolist()
        return predict, scores

    def on_iter_end(self, step, writer):
        most_common = [desc for desc, count in self.word_counts.most_common(10)]
        insts = [instance.Instance(input=desc) for desc in most_common]
        xs, y = self._data_to_arrays(insts, test=True)
        probs = self.model.predict(xs)
        for i, desc in enumerate(most_common):
            dist = probs[i, :]
            for image, channel in zip(self.color_vec.visualize_distribution(dist), '012'):
                writer.log_image(step, 'listener/%s/%s' % (desc, channel), image)
        super(ListenerLearner, self).on_iter_end(step, writer)

    def _data_to_arrays(self, training_instances, test=False):
        if not test:
            self.seq_vec.add_all(['<s>'] + inst.input.split() + ['</s>']
                                 for inst in training_instances)

        sentences = []
        colors = []
        for i, inst in enumerate(training_instances):
            self.word_counts.update([inst.input])
            desc = inst.input.split()
            if inst.output:
                (hue, sat, val) = inst.output
            else:
                assert test
                hue = sat = val = 0.0
            color_0_1 = colorsys.hsv_to_rgb(hue / 360.0, sat / 100.0, val / 100.0)
            color = tuple(min(d * 256, 255) for d in color_0_1)
            s = ['<s>'] * (self.seq_vec.max_len - 1 - len(desc)) + desc
            s.append('</s>')
            # print('%s -> %s' % (repr(s), repr(color)))
            sentences.append(s)
            colors.append(color)
        print('Number of sequences: %d' % len(sentences))

        print('Vectorization')
        x = np.zeros((len(sentences), self.seq_vec.max_len), dtype=np.int32)
        y = np.zeros((len(sentences),), dtype=np.int32)
        for i, sentence in enumerate(sentences):
            x[i, :] = self.seq_vec.vectorize(sentence)
            y[i] = self.color_vec.vectorize(colors[i])

        return [x], [y]

    def log_prior_emp(self, input_vars):
        raise NotImplementedError

    def log_prior_smooth(self, input_vars):
        # TODO
        return self.log_prior_emp(input_vars)

    def sample(self, inputs):
        raise NotImplementedError

    def _build_model(self, model_class=SimpleLasagneModel):
        input_var = T.imatrix('inputs')
        target_var = T.ivector('targets')

        l_out = self._get_l_out([input_var])

        self.model = model_class([input_var], [target_var], l_out,
                                 loss=categorical_crossentropy, optimizer=rmsprop)

    def _get_l_out(self, input_vars):
        options = config.options()

        input_var = input_vars[0]

        l_in = InputLayer(shape=(None, self.seq_vec.max_len), input_var=input_var)
        l_in_embed = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                                    output_size=options.listener_cell_size)
        l_lstm1 = LSTMLayer(l_in_embed, num_units=options.listener_cell_size,
                            forgetgate=Gate(b=Constant(options.listener_forget_bias)))
        l_lstm1_drop = DropoutLayer(l_lstm1, p=0.2)
        l_lstm2 = LSTMLayer(l_lstm1_drop, num_units=options.listener_cell_size,
                            forgetgate=Gate(b=Constant(options.listener_forget_bias)))
        l_lstm2_drop = DropoutLayer(l_lstm2, p=0.2)

        l_hidden = DenseLayer(l_lstm2_drop, num_units=options.listener_cell_size, nonlinearity=None)
        l_hidden_drop = DropoutLayer(l_hidden, p=0.2)
        l_scores = DenseLayer(l_hidden_drop, num_units=self.color_vec.num_types, nonlinearity=None)
        return NonlinearityLayer(l_scores, nonlinearity=softmax)
