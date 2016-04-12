import numpy as np
import theano
import theano.tensor as T
import warnings
from collections import Counter
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, EmbeddingLayer, NonlinearityLayer
from lasagne.layers.recurrent import Gate
from lasagne.init import Constant
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from lasagne.updates import rmsprop

from stanza.unstable import config, instance, progress, iterators, rng
from neural import NeuralLearner, SimpleLasagneModel
from neural import NONLINEARITIES, OPTIMIZERS, CELLS, sample
from vectorizers import SequenceVectorizer, BucketsVectorizer, SymbolVectorizer

random = rng.get_rng()

parser = config.get_options_parser()
parser.add_argument('--listener_cell_size', type=int, default=20,
                    help='The number of dimensions of all hidden layers and cells in '
                         'the listener model. If 0 and using the AtomicListenerLearner, '
                         'remove all hidden layers and only train a linear classifier.')
parser.add_argument('--listener_forget_bias', type=float, default=5.0,
                    help='The initial value of the forget gate bias in LSTM cells in '
                         'the listener model. A positive initial forget gate bias '
                         'encourages the model to remember everything by default.')
parser.add_argument('--listener_nonlinearity', choices=NONLINEARITIES.keys(), default='tanh',
                    help='The nonlinearity/activation function to use for dense and '
                         'LSTM layers in the listener model.')
parser.add_argument('--listener_cell', choices=CELLS.keys(), default='LSTM',
                    help='The recurrent cell to use for the listener model.')
parser.add_argument('--listener_dropout', type=float, default=0.2,
                    help='The dropout rate (probability of setting a value to zero). '
                         'Dropout will be disabled if nonpositive.')
parser.add_argument('--listener_color_resolution', type=int, nargs='+', default=[4],
                    help='The number of buckets along each dimension of color space '
                         'for the output of the listener model.')
parser.add_argument('--listener_hsv', type=config.boolean, default=False,
                    help='If True, output color buckets are in HSV space; otherwise, '
                         'color buckets will be in RGB. Final output instances will be in HSV '
                         'regardless; this sets the internal representation for training '
                         'and prediction.')
parser.add_argument('--listener_eval_batch_size', type=int, default=65536,
                    help='The number of examples per batch for evaluating the listener '
                         'model. Higher means faster but more memory usage. This should '
                         'not affect modeling accuracy.')
parser.add_argument('--listener_optimizer', choices=OPTIMIZERS.keys(), default='rmsprop',
                    help='The optimization (update) algorithm to use for listener training.')
parser.add_argument('--listener_learning_rate', type=float, default=1.0,
                    help='The learning rate to use for listener training.')
parser.add_argument('--listener_grad_clipping', type=float, default=0.0,
                    help='The maximum absolute value of the gradient messages for the'
                         'LSTM component of the listener model.')


class UnigramPrior(object):
    def __init__(self, vocab_size, mask_index=None):
        self.vocab_size = vocab_size
        self.counts = theano.shared(np.zeros((vocab_size,), dtype=np.int32))
        self.total = theano.shared(np.array(0, dtype=np.int32))
        self.mask_index = mask_index
        self.log_probs = T.cast(self.counts, 'float32') / T.cast(self.total, 'float32')

    def fit(self, xs, ys):
        (x,) = xs
        counts = np.bincount(x.flatten(), minlength=self.vocab_size).astype(np.int32)
        if self.mask_index is not None:
            counts[self.mask_index] = 0
        self.counts.set_value(self.counts.get_value() + counts)
        self.total.set_value(self.total.get_value() + np.sum(counts))

    def apply(self, input_vars):
        (x,) = input_vars

        token_probs = self.log_probs[x]
        if self.mask_index is not None:
            token_probs = token_probs * T.cast((x == self.mask_index), 'float32')
        if token_probs.ndim == 1:
            return token_probs
        else:
            return token_probs.sum(axis=1)

    def sample(self, num_samples=1):
        return np.array([sample(self.counts.get_value() * 1.0 / self.total.get_value())
                         for _ in range(num_samples)], dtype=np.int32)


class AtomicUniformPrior(object):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def fit(self, xs, ys):
        pass

    def apply(self, input_vars):
        c = input_vars[0]
        if c.ndim == 1:
            ones = T.ones_like(c)
        else:
            ones = T.ones_like(c[:, 0])
        return -np.log(self.vocab_size) * ones

    def sample(self, num_samples=1):
        return random.randint(0, self.vocab_size, size=(num_samples,))


PRIORS = {
    'Unigram': UnigramPrior,
    'AtomicUniform': AtomicUniformPrior,
}

parser.add_argument('--listener_prior', choices=PRIORS.keys(), default='Unigram',
                    help='The prior model for the listener (prior over utterances). '
                         'Only used in RSA learner.')


class ListenerLearner(NeuralLearner):
    '''
    An LSTM-based listener (guesses colors from descriptions).
    '''
    def __init__(self, id=None):
        super(ListenerLearner, self).__init__(id=id)
        options = config.options()
        self.word_counts = Counter()
        self.seq_vec = SequenceVectorizer()
        self.color_vec = BucketsVectorizer(options.speaker_color_resolution,
                                           hsv=options.speaker_hsv)

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        options = config.options()

        predictions = []
        scores = []
        batches = iterators.iter_batches(eval_instances, options.listener_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // options.listener_eval_batch_size + 1

        if options.verbosity + verbosity >= 2:
            print('Testing')
        progress.start_task('Eval batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)

            xs, (y,) = self._data_to_arrays(batch, test=True)

            probs = self.model.predict(xs)
            if random:
                indices = sample(probs)
                predictions.extend(self.color_vec.unvectorize_all(indices,
                                                                  random=True, hsv=True))
            else:
                predictions.extend(self.color_vec.unvectorize_all(probs.argmax(axis=1),
                                                                  hsv=True))
            bucket_volume = (256.0 ** 3) / self.color_vec.num_types
            scores_arr = np.log(probs[np.arange(len(batch)), y]) - np.log(bucket_volume)
            scores.extend(scores_arr.tolist())
        progress.end_task()
        if options.verbosity >= 9:
            print('%s %ss:') % (self.id, 'sample' if random else 'prediction')
            for inst, prediction in zip(eval_instances, predictions):
                print('%s -> %s' % (repr(inst.input), repr(prediction)))

        return predictions, scores

    def on_iter_end(self, step, writer):
        most_common = [desc for desc, count in self.word_counts.most_common(10)]
        insts = [instance.Instance(input=desc) for desc in most_common]
        xs, (y,) = self._data_to_arrays(insts, test=True)
        probs = self.model.predict(xs)
        for i, desc in enumerate(most_common):
            dist = probs[i, :]
            for image, channel in zip(self.color_vec.visualize_distribution(dist), '012'):
                writer.log_image(step, '%s/%s/%s' % (self.id, desc, channel), image)
        super(ListenerLearner, self).on_iter_end(step, writer)

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        options = config.options()

        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_desc, get_color = (get_o, get_i) if inverted else (get_i, get_o)

        if init_vectorizer:
            self.seq_vec.add_all(['<s>'] + get_desc(inst).split() + ['</s>']
                                 for inst in training_instances)
            self.word_counts.update([get_desc(inst) for inst in training_instances])

        sentences = []
        colors = []
        if options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        for i, inst in enumerate(training_instances):
            desc = get_desc(inst).split()
            color = get_color(inst)
            if not color:
                assert test
                color = (0.0, 0.0, 0.0)
            s = ['<s>'] * (self.seq_vec.max_len - 1 - len(desc)) + desc
            s.append('</s>')
            if options.verbosity >= 9:
                print('%s -> %s' % (repr(s), repr(color)))
            sentences.append(s)
            colors.append(color)

        x = np.zeros((len(sentences), self.seq_vec.max_len), dtype=np.int32)
        y = np.zeros((len(sentences),), dtype=np.int32)
        for i, sentence in enumerate(sentences):
            x[i, :] = self.seq_vec.vectorize(sentence)
            y[i] = self.color_vec.vectorize(colors[i], hsv=True)

        return [x], [y]

    def _build_model(self, model_class=SimpleLasagneModel):
        options = config.options()
        id_tag = (self.id + '/') if self.id else ''

        input_var = T.imatrix(id_tag + 'inputs')
        target_var = T.ivector(id_tag + 'targets')

        self.l_out, self.input_layers = self._get_l_out([input_var])
        self.loss = categorical_crossentropy

        self.model = model_class([input_var], [target_var], self.l_out,
                                 loss=self.loss, optimizer=OPTIMIZERS[options.listener_optimizer],
                                 learning_rate=options.listener_learning_rate,
                                 id=self.id)

        prior_class = PRIORS[options.listener_prior]
        self.prior_emp = prior_class(vocab_size=len(self.seq_vec.tokens))
        self.prior_smooth = prior_class(vocab_size=len(self.seq_vec.tokens))  # TODO: smoothing

    def _get_l_out(self, input_vars):
        options = config.options()
        check_options(options)
        id_tag = (self.id + '/') if self.id else ''

        input_var = input_vars[0]

        l_in = InputLayer(shape=(None, self.seq_vec.max_len), input_var=input_var,
                          name=id_tag + 'desc_input')
        l_in_embed = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                                    output_size=options.listener_cell_size,
                                    name=id_tag + 'desc_embed')

        cell = CELLS[options.speaker_cell]
        cell_kwargs = {
            'grad_clipping': options.speaker_grad_clipping,
            'num_units': options.listener_cell_size,
        }
        if options.speaker_cell == 'LSTM':
            cell_kwargs['forgetgate'] = Gate(b=Constant(options.speaker_forget_bias))
        if options.speaker_cell != 'GRU':
            cell_kwargs['nonlinearity'] = NONLINEARITIES[options.speaker_nonlinearity]

        l_rec1 = cell(l_in_embed, name=id_tag + 'rec1', **cell_kwargs)
        if options.listener_dropout > 0.0:
            l_rec1_drop = DropoutLayer(l_rec1, p=options.listener_dropout,
                                       name=id_tag + 'rec1_drop')
        else:
            l_rec1_drop = l_rec1
        l_rec2 = cell(l_rec1_drop, name=id_tag + 'rec2', **cell_kwargs)
        if options.listener_dropout > 0.0:
            l_rec2_drop = DropoutLayer(l_rec2, p=options.listener_dropout,
                                       name=id_tag + 'rec2_drop')
        else:
            l_rec2_drop = l_rec2

        l_hidden = DenseLayer(l_rec2_drop, num_units=options.listener_cell_size,
                              nonlinearity=NONLINEARITIES[options.listener_nonlinearity],
                              name=id_tag + 'hidden')
        if options.listener_dropout > 0.0:
            l_hidden_drop = DropoutLayer(l_hidden, p=options.listener_dropout,
                                         name=id_tag + 'hidden_drop')
        else:
            l_hidden_drop = l_hidden
        l_scores = DenseLayer(l_hidden_drop, num_units=self.color_vec.num_types, nonlinearity=None,
                              name=id_tag + 'scores')
        l_out = NonlinearityLayer(l_scores, nonlinearity=softmax, name=id_tag + 'out')

        return l_out, [l_in]

    def sample_prior_smooth(self, num_samples):
        return [instance.Instance(input=c) for c in
                self.seq_vec.unvectorize_all(self.prior_smooth.sample(num_samples))]


class AtomicListenerLearner(ListenerLearner):
    '''
    An single-embedding listener (guesses colors from descriptions, where
    the descriptions are treated as indivisible symbols).
    '''
    def __init__(self, id=None):
        super(AtomicListenerLearner, self).__init__(id=id)
        self.seq_vec = SymbolVectorizer()

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        options = config.options()

        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_desc, get_color = (get_o, get_i) if inverted else (get_i, get_o)

        if init_vectorizer:
            self.seq_vec.add_all(get_desc(inst) for inst in training_instances)

        sentences = []
        colors = []
        if options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        for i, inst in enumerate(training_instances):
            self.word_counts.update([get_desc(inst)])
            desc = get_desc(inst)
            color = get_color(inst)
            if not color:
                assert test
                color = (0.0, 0.0, 0.0)
            if options.verbosity >= 9:
                print('%s -> %s' % (repr(desc), repr(color)))
            sentences.append(desc)
            colors.append(color)

        x = np.zeros((len(sentences),), dtype=np.int32)
        y = np.zeros((len(sentences),), dtype=np.int32)
        for i, sentence in enumerate(sentences):
            x[i] = self.seq_vec.vectorize(sentence)
            y[i] = self.color_vec.vectorize(colors[i], hsv=True)

        return [x], [y]

    def _build_model(self, model_class=SimpleLasagneModel):
        options = config.options()

        id_tag = (self.id + '/') if self.id else ''
        input_var = T.ivector(id_tag + 'inputs')
        target_var = T.ivector(id_tag + 'targets')

        self.l_out, self.input_layers = self._get_l_out([input_var])
        self.loss = categorical_crossentropy

        self.model = model_class([input_var], [target_var], self.l_out,
                                 loss=self.loss, optimizer=rmsprop, id=self.id)

        prior_class = PRIORS[options.listener_prior]
        self.prior_emp = prior_class(vocab_size=len(self.seq_vec.tokens))
        self.prior_smooth = prior_class(vocab_size=len(self.seq_vec.tokens))  # TODO: smoothing

    def _get_l_out(self, input_vars):
        options = config.options()
        id_tag = (self.id + '/') if self.id else ''

        input_var = input_vars[0]

        l_in = InputLayer(shape=(None,), input_var=input_var,
                          name=id_tag + 'desc_input')
        embed_size = options.listener_cell_size or self.color_vec.num_types
        l_in_embed = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                                    output_size=embed_size,
                                    name=id_tag + 'desc_embed')

        if options.listener_cell_size == 0:
            l_scores = l_in_embed  # BiasLayer(l_in_embed, name=id_tag + 'bias')
        else:
            l_hidden = DenseLayer(l_in_embed, num_units=options.listener_cell_size,
                                  nonlinearity=NONLINEARITIES[options.listener_nonlinearity],
                                  name=id_tag + 'hidden')
            if options.listener_dropout > 0.0:
                l_hidden_drop = DropoutLayer(l_hidden, p=options.listener_dropout,
                                             name=id_tag + 'hidden_drop')
            else:
                l_hidden_drop = l_hidden

            l_scores = DenseLayer(l_hidden_drop, num_units=self.color_vec.num_types,
                                  nonlinearity=None, name=id_tag + 'scores')
        l_out = NonlinearityLayer(l_scores, nonlinearity=softmax, name=id_tag + 'out')

        return l_out, [l_in]

    def sample_prior_smooth(self, num_samples):
        return [instance.Instance(input=c) for c in
                self.seq_vec.unvectorize_all(self.prior_smooth.sample(num_samples))]


def check_options(options):
    if options.listener_grad_clipping:
        warnings.warn('Per-dimension gradient clipping (--listener_grad_clipping) is enabled. '
                      'This feature is unlikely to correctly constrain gradients and avoid '
                      'NaNs; use --true_grad_clipping instead.')
    if not options.true_grad_clipping:
        warnings.warn('Norm-constraint gradient clipping is disabled for a recurrent model. '
                      'This will likely lead to exploding gradients.')
    if options.true_grad_clipping > 6.0:
        warnings.warn('Gradient clipping norm is unusually high (%s). '
                      'This could lead to exploding gradients.' % options.true_grad_clipping)
    if options.listener_nonlinearity == 'rectify':
        warnings.warn('Using ReLU as the output nonlinearity for a recurrent unit. This may '
                      'be a source of NaNs in the gradient.')


LISTENERS = {
    'Listener': ListenerLearner,
    'AtomicListener': AtomicListenerLearner,
}
