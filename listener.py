import numpy as np
import theano
import theano.tensor as T
import warnings
from collections import Counter
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, EmbeddingLayer, NonlinearityLayer
from lasagne.layers import NINLayer, ConcatLayer, dimshuffle
from lasagne.layers.recurrent import Gate
from lasagne.init import Constant
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from lasagne.updates import rmsprop

from stanza.monitoring import progress
from stanza.research import config, instance, iterators, rng
import color_instances
import speaker
from neural import NeuralLearner, SimpleLasagneModel
from neural import NONLINEARITIES, OPTIMIZERS, CELLS, sample
from vectorizers import SequenceVectorizer, BucketsVectorizer, SymbolVectorizer
from vectorizers import strip_invalid_tokens, COLOR_REPRS

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
parser.add_argument('--listener_hidden_color_layers', type=int, default=0,
                    help='The number of dense layers after the color representation '
                         '(ContextListenerLearner only).')
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
parser.add_argument('--listener_learning_rate', type=float, default=0.1,
                    help='The learning rate to use for listener training.')
parser.add_argument('--listener_grad_clipping', type=float, default=0.0,
                    help='The maximum absolute value of the gradient messages for the'
                         'LSTM component of the listener model.')
parser.add_argument('--listener_color_repr', choices=COLOR_REPRS.keys(), default='buckets',
                    help='The representation of the color to use in the listener model: a regular '
                         'grid of `buckets`, overlapping bucket grids at multiple resolutions '
                         '(`ms`), `raw` RGB/HSV values, or a `fourier` transform-based '
                         'representation. Only used for ContextListenerLearner.')


class UnigramPrior(object):
    '''
    >>> p = UnigramPrior()
    >>> p.train([instance.Instance('blue')])
    >>> p.sample(3)  # doctest: +ELLIPSIS
    [Instance('...', None), Instance('...', None), Instance('...', None)]
    '''
    def __init__(self):
        self.vec = SequenceVectorizer()
        self.vec.add_all([['</s>'], ['<MASK>']])
        self.counts = theano.shared(np.zeros((self.vec.num_types,), dtype=np.int32))
        self.total = theano.shared(np.array(0, dtype=np.int32))
        self.log_probs = T.cast(self.counts, 'float32') / T.cast(self.total, 'float32')
        self.mask_index = self.vec.vectorize(['<MASK>'])[0]

    def train(self, training_instances, listener_data=True):
        get_utt = (lambda inst: inst.input) if listener_data else (lambda inst: inst.output)
        tokenized = [get_utt(inst).split() for inst in training_instances]
        self.vec.add_all(tokenized)
        x = self.vec.vectorize_all(self.pad(tokenized, self.vec.max_len))
        vocab_size = self.vec.num_types

        counts = np.bincount(x.flatten(), minlength=vocab_size).astype(np.int32)
        counts[self.mask_index] = 0
        self.counts.set_value(counts)
        self.total.set_value(np.sum(counts))

    def apply(self, input_vars):
        (x,) = input_vars

        token_probs = self.log_probs[x]
        if self.mask_index is not None:
            token_probs = token_probs * T.cast(T.eq(x, self.mask_index), 'float32')
        if token_probs.ndim == 1:
            return token_probs
        else:
            return token_probs.sum(axis=1)

    def sample(self, num_samples=1):
        indices = np.array([[sample(self.counts.get_value() * 1.0 / self.total.get_value())
                             for _t in range(self.vec.max_len)]
                            for _s in range(num_samples)], dtype=np.int32)
        return [instance.Instance(' '.join(strip_invalid_tokens(s)))
                for s in self.vec.unvectorize_all(indices)]

    def pad(self, sequences, length):
        '''
        Adds </s> tokens followed by zero or more <MASK> tokens to bring the total
        length of all sequences to `length + 1` (the addition of one is because all
        sequences receive a </s>, but `length` should be the max length of the original
        sequences).

        >>> UnigramPrior().pad([['blue'], ['very', 'blue']], 2)
        [['blue', '</s>', '<MASK>'], ['very', 'blue', '</s>']]
        '''
        return [seq + ['</s>'] + ['<MASK>'] * (length - len(seq))
                for seq in sequences]


class AtomicUniformPrior(object):
    '''
    >>> p = AtomicUniformPrior()
    >>> p.train([instance.Instance('blue')])
    >>> p.sample(3)  # doctest: +ELLIPSIS
    [Instance('...', None), Instance('...', None), Instance('...', None)]
    '''
    def __init__(self):
        self.vec = SymbolVectorizer()

    def train(self, training_instances, listener_data=True):
        self.vec.add_all([inst.input if listener_data else inst.output
                          for inst in training_instances])

    def apply(self, input_vars):
        c = input_vars[0]
        if c.ndim == 1:
            ones = T.ones_like(c)
        else:
            ones = T.ones_like(c[:, 0])
        return -np.log(self.vec.num_types) * ones

    def sample(self, num_samples=1):
        indices = random.randint(0, self.vec.num_types, size=(num_samples,))
        return [instance.Instance(c) for c in self.vec.unvectorize_all(indices)]


class UnigramContextPrior(UnigramPrior):
    def __init__(self):
        super(UnigramContextPrior, self).__init__()
        self.uniform_colors = speaker.UniformPrior()

    def apply(self, input_vars):
        options = config.options()
        context_len = options.num_distractors + 1
        return (super(UnigramContextPrior, self).apply(input_vars) -
                3.0 * np.log(256.0) * context_len)

    def sample(self, num_samples=1):
        descs = super(UnigramContextPrior, self).sample(num_samples=num_samples)
        colors = self.uniform_colors.sample(num_samples)
        insts = [instance.Instance(d.input, c.input) for d, c in zip(descs, colors)]
        return color_instances.reference_game(insts, color_instances.uniform, listener=True)


class AtomicUniformContextPrior(AtomicUniformPrior):
    def __init__(self):
        super(AtomicUniformContextPrior, self).__init__()
        self.uniform_colors = speaker.UniformPrior()

    def apply(self, input_vars):
        options = config.options()
        context_len = options.num_distractors + 1
        return (super(AtomicUniformContextPrior, self).apply(input_vars) -
                3.0 * np.log(256.0) * context_len)

    def sample(self, num_samples=1):
        descs = super(AtomicUniformContextPrior, self).sample(num_samples=num_samples)
        colors = self.uniform_colors.sample(num_samples)
        insts = [instance.Instance(d.input, c.input) for d, c in zip(descs, colors)]
        return color_instances.reference_game(insts, color_instances.uniform, listener=True)


PRIORS = {
    'Unigram': UnigramPrior,
    'AtomicUniform': AtomicUniformPrior,
    'UnigramContext': UnigramContextPrior,
    'AtomicUniformContext': AtomicUniformContextPrior,
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
        self.word_counts = Counter()
        self.seq_vec = SequenceVectorizer()
        self.color_vec = BucketsVectorizer(self.options.listener_color_resolution,
                                           hsv=self.options.listener_hsv)

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        predictions = []
        scores = []
        batches = iterators.iter_batches(eval_instances, self.options.listener_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // self.options.listener_eval_batch_size + 1

        if self.options.verbosity + verbosity >= 2:
            print('Testing')
        progress.start_task('Eval batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)

            xs, (y,) = self._data_to_arrays(batch, test=True)

            probs = self.model.predict(xs)
            if random:
                indices = sample(probs)
                predictions.extend(self.unvectorize(indices, random=True))
            else:
                predictions.extend(self.unvectorize(probs.argmax(axis=1)))
            scores_arr = np.log(probs[np.arange(len(batch)), y]) + self.bucket_adjustment()
            scores.extend(scores_arr.tolist())
        progress.end_task()
        if self.options.verbosity >= 9:
            print('%s %ss:') % (self.id, 'sample' if random else 'prediction')
            for inst, prediction in zip(eval_instances, predictions):
                print('%s -> %s' % (repr(inst.input), repr(prediction)))

        return predictions, scores

    def unvectorize(self, indices, random=False):
        return self.color_vec.unvectorize_all(indices, random=random, hsv=True)

    def bucket_adjustment(self):
        bucket_volume = (256.0 ** 3) / self.color_vec.num_types
        return -np.log(bucket_volume)

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
        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_desc, get_color = (get_o, get_i) if inverted else (get_i, get_o)

        if init_vectorizer:
            self.seq_vec.add_all(['<s>'] + get_desc(inst).split() + ['</s>']
                                 for inst in training_instances)
            self.word_counts.update([get_desc(inst) for inst in training_instances])

        sentences = []
        colors = []
        if self.options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        for i, inst in enumerate(training_instances):
            desc = get_desc(inst).split()
            color = get_color(inst)
            if not color:
                assert test
                color = (0.0, 0.0, 0.0)
            s = ['<s>'] * (self.seq_vec.max_len - 1 - len(desc)) + desc
            s.append('</s>')
            if self.options.verbosity >= 9:
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
        id_tag = (self.id + '/') if self.id else ''

        input_var = T.imatrix(id_tag + 'inputs')
        target_var = T.ivector(id_tag + 'targets')

        self.l_out, self.input_layers = self._get_l_out([input_var])
        self.loss = categorical_crossentropy

        self.model = model_class(
            [input_var], [target_var], self.l_out,
            loss=self.loss, optimizer=OPTIMIZERS[self.options.listener_optimizer],
            learning_rate=self.options.listener_learning_rate,
            id=self.id)

    def train_priors(self, training_instances, listener_data=False):
        prior_class = PRIORS[self.options.listener_prior]
        self.prior_emp = prior_class()  # TODO: accurate values for empirical prior
        self.prior_smooth = prior_class()

        self.prior_emp.train(training_instances, listener_data=listener_data)
        self.prior_smooth.train(training_instances, listener_data=listener_data)

    def _get_l_out(self, input_vars):
        check_options(self.options)
        id_tag = (self.id + '/') if self.id else ''

        input_var = input_vars[0]

        l_in = InputLayer(shape=(None, self.seq_vec.max_len), input_var=input_var,
                          name=id_tag + 'desc_input')
        l_in_embed = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                                    output_size=self.options.listener_cell_size,
                                    name=id_tag + 'desc_embed')

        cell = CELLS[self.options.listener_cell]
        cell_kwargs = {
            'grad_clipping': self.options.listener_grad_clipping,
            'num_units': self.options.listener_cell_size,
        }
        if self.options.listener_cell == 'LSTM':
            cell_kwargs['forgetgate'] = Gate(b=Constant(self.options.listener_forget_bias))
        if self.options.listener_cell != 'GRU':
            cell_kwargs['nonlinearity'] = NONLINEARITIES[self.options.listener_nonlinearity]

        l_rec1 = cell(l_in_embed, name=id_tag + 'rec1', **cell_kwargs)
        if self.options.listener_dropout > 0.0:
            l_rec1_drop = DropoutLayer(l_rec1, p=self.options.listener_dropout,
                                       name=id_tag + 'rec1_drop')
        else:
            l_rec1_drop = l_rec1
        l_rec2 = cell(l_rec1_drop, name=id_tag + 'rec2', **cell_kwargs)
        if self.options.listener_dropout > 0.0:
            l_rec2_drop = DropoutLayer(l_rec2, p=self.options.listener_dropout,
                                       name=id_tag + 'rec2_drop')
        else:
            l_rec2_drop = l_rec2

        l_hidden = DenseLayer(l_rec2_drop, num_units=self.options.listener_cell_size,
                              nonlinearity=NONLINEARITIES[self.options.listener_nonlinearity],
                              name=id_tag + 'hidden')
        if self.options.listener_dropout > 0.0:
            l_hidden_drop = DropoutLayer(l_hidden, p=self.options.listener_dropout,
                                         name=id_tag + 'hidden_drop')
        else:
            l_hidden_drop = l_hidden
        l_scores = DenseLayer(l_hidden_drop, num_units=self.color_vec.num_types, nonlinearity=None,
                              name=id_tag + 'scores')
        l_out = NonlinearityLayer(l_scores, nonlinearity=softmax, name=id_tag + 'out')

        return l_out, [l_in]

    def sample_prior_smooth(self, num_samples):
        return self.prior_smooth.sample(num_samples)


class ContextListenerLearner(ListenerLearner):
    def __init__(self, *args, **kwargs):
        super(ContextListenerLearner, self).__init__(*args, **kwargs)

        color_repr = COLOR_REPRS[self.options.listener_color_repr]
        self.color_vec = color_repr(self.options.listener_color_resolution,
                                    hsv=self.options.listener_hsv)

    @property
    def recurrent_context(self):
        return True

    @property
    def context_len(self):
        return self.options.num_distractors + 1

    def unvectorize(self, indices, random=False):
        return indices

    def bucket_adjustment(self):
        return 0.0

    def on_iter_end(self, step, writer):
        pass

    def _build_model(self, model_class=SimpleLasagneModel):
        id_tag = (self.id + '/') if self.id else ''

        input_var = T.imatrix(id_tag + 'inputs')
        context_vars = self.color_vec.get_input_vars(self.id, recurrent=self.recurrent_context)
        target_var = T.ivector(id_tag + 'targets')

        self.l_out, self.input_layers = self._get_l_out([input_var] + context_vars)
        self.loss = categorical_crossentropy

        self.model = model_class(
            [input_var] + context_vars, [target_var], self.l_out,
            loss=self.loss, optimizer=OPTIMIZERS[self.options.listener_optimizer],
            learning_rate=self.options.listener_learning_rate,
            id=self.id)

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_desc, get_color_index = (get_o, get_i) if inverted else (get_i, get_o)
        get_alt_i, get_alt_o = (lambda inst: inst.alt_inputs), (lambda inst: inst.alt_outputs)
        get_alt_colors = get_alt_i if inverted else get_alt_o

        if init_vectorizer:
            self.seq_vec.add_all(['<s>'] + get_desc(inst).split() + ['</s>']
                                 for inst in training_instances)
            self.word_counts.update([get_desc(inst) for inst in training_instances])

        sentences = []
        colors = []
        target_indices = []
        if self.options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        for i, inst in enumerate(training_instances):
            desc = get_desc(inst).split()
            target = get_color_index(inst)
            if target is None:
                assert test
                target = 0
            s = ['<s>'] * (self.seq_vec.max_len - 1 - len(desc)) + desc
            s.append('</s>')
            new_context = get_alt_colors(inst)
            assert len(new_context) == self.context_len, \
                'Inconsistent context lengths: %s' % ((self.context_len, len(new_context)),)
            if self.options.verbosity >= 9:
                print('%s [%s] -> %s' % (repr(s), repr(new_context), repr(target)))
            sentences.append(s)
            target_indices.append(target)
            colors.extend(new_context)

        x = np.zeros((len(sentences), self.seq_vec.max_len), dtype=np.int32)
        for i, sentence in enumerate(sentences):
            x[i, :] = self.seq_vec.vectorize(sentence)
        y = np.array(target_indices, dtype=np.int32)

        c = self.color_vec.vectorize_all(colors, hsv=True)
        if len(c.shape) == 1:
            c = c.reshape((len(colors) / self.context_len, self.context_len))
        else:
            c = c.reshape((len(colors) / self.context_len, self.context_len * c.shape[1]) +
                          c.shape[2:])
        if self.recurrent_context:
            c = np.tile(c[:, np.newaxis, ...], [1, self.seq_vec.max_len] + [1] * (c.ndim - 1))

        if self.options.verbosity >= 9:
            print('x: %s' % (repr(x),))
            print('c: %s' % (repr(c),))
            print('y: %s' % (repr(y),))
        return [x, c], [y]

    def _get_l_out(self, input_vars):
        check_options(self.options)
        id_tag = (self.id + '/') if self.id else ''

        input_var = input_vars[0]
        context_vars = input_vars[1:]

        l_in = InputLayer(shape=(None, self.seq_vec.max_len), input_var=input_var,
                          name=id_tag + 'desc_input')
        l_in_embed = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                                    output_size=self.options.listener_cell_size,
                                    name=id_tag + 'desc_embed')

        # Context repr has shape (batch_size, seq_len, context_len * repr_size)
        l_context_repr, context_inputs = self.color_vec.get_input_layer(
            context_vars,
            recurrent_length=self.seq_vec.max_len,
            cell_size=self.options.listener_cell_size,
            context_len=self.context_len,
            id=self.id
        )
        l_hidden_context = dimshuffle(l_context_repr, (0, 2, 1))
        for i in range(1, self.options.listener_hidden_color_layers + 1):
            l_hidden_context = NINLayer(
                l_hidden_context, num_units=self.options.listener_cell_size,
                nonlinearity=NONLINEARITIES[self.options.listener_nonlinearity],
                name=id_tag + 'hidden_context%d' % i)
        l_hidden_context = dimshuffle(l_hidden_context, (0, 2, 1))
        l_concat = ConcatLayer([l_hidden_context, l_in_embed], axis=2,
                               name=id_tag + 'concat_inp_context')

        cell = CELLS[self.options.listener_cell]
        cell_kwargs = {
            'grad_clipping': self.options.listener_grad_clipping,
            'num_units': self.options.listener_cell_size,
        }
        if self.options.listener_cell == 'LSTM':
            cell_kwargs['forgetgate'] = Gate(b=Constant(self.options.listener_forget_bias))
        if self.options.listener_cell != 'GRU':
            cell_kwargs['nonlinearity'] = NONLINEARITIES[self.options.listener_nonlinearity]

        l_rec1 = cell(l_concat, name=id_tag + 'rec1', **cell_kwargs)
        if self.options.listener_dropout > 0.0:
            l_rec1_drop = DropoutLayer(l_rec1, p=self.options.listener_dropout,
                                       name=id_tag + 'rec1_drop')
        else:
            l_rec1_drop = l_rec1
        l_rec2 = cell(l_rec1_drop, name=id_tag + 'rec2', **cell_kwargs)
        if self.options.listener_dropout > 0.0:
            l_rec2_drop = DropoutLayer(l_rec2, p=self.options.listener_dropout,
                                       name=id_tag + 'rec2_drop')
        else:
            l_rec2_drop = l_rec2

        l_hidden = DenseLayer(l_rec2_drop, num_units=self.options.listener_cell_size,
                              nonlinearity=NONLINEARITIES[self.options.listener_nonlinearity],
                              name=id_tag + 'hidden')
        if self.options.listener_dropout > 0.0:
            l_hidden_drop = DropoutLayer(l_hidden, p=self.options.listener_dropout,
                                         name=id_tag + 'hidden_drop')
        else:
            l_hidden_drop = l_hidden
        l_scores = DenseLayer(l_hidden_drop, num_units=self.context_len, nonlinearity=softmax,
                              name=id_tag + 'scores')

        return l_scores, [l_in] + context_inputs


class TwoStreamListenerLearner(ContextListenerLearner):
    def __init__(self, *args, **kwargs):
        super(TwoStreamListenerLearner, self).__init__(*args, **kwargs)

    @property
    def recurrent_context(self):
        return False

    def _get_l_out(self, input_vars):
        check_options(self.options)
        id_tag = (self.id + '/') if self.id else ''

        input_var = input_vars[0]
        context_vars = input_vars[1:]

        l_in = InputLayer(shape=(None, self.seq_vec.max_len), input_var=input_var,
                          name=id_tag + 'desc_input')
        l_in_embed = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                                    output_size=self.options.listener_cell_size,
                                    name=id_tag + 'desc_embed')

        cell = CELLS[self.options.listener_cell]
        cell_kwargs = {
            'grad_clipping': self.options.listener_grad_clipping,
            'num_units': self.options.listener_cell_size,
        }
        if self.options.listener_cell == 'LSTM':
            cell_kwargs['forgetgate'] = Gate(b=Constant(self.options.listener_forget_bias))
        if self.options.listener_cell != 'GRU':
            cell_kwargs['nonlinearity'] = NONLINEARITIES[self.options.listener_nonlinearity]

        l_rec1 = cell(l_in_embed, name=id_tag + 'rec1', only_return_final=True, **cell_kwargs)
        if self.options.listener_dropout > 0.0:
            l_rec1_drop = DropoutLayer(l_rec1, p=self.options.listener_dropout,
                                       name=id_tag + 'rec1_drop')
        else:
            l_rec1_drop = l_rec1
        '''
        l_rec2 = cell(l_rec1_drop, name=id_tag + 'rec2', only_return_final=True, **cell_kwargs)
        if self.options.listener_dropout > 0.0:
            l_rec2_drop = DropoutLayer(l_rec2, p=self.options.listener_dropout,
                                       name=id_tag + 'rec2_drop')
        else:
            l_rec2_drop = l_rec2
        '''
        # remove only_return_final from l_rec1 to restore second layer
        l_rec2_drop = l_rec1_drop

        # Context repr has shape (batch_size, context_len * repr_size)
        l_context_repr, context_inputs = self.color_vec.get_input_layer(
            context_vars,
            cell_size=self.options.listener_cell_size,
            context_len=self.context_len,
            id=self.id
        )
        l_concat = ConcatLayer([l_context_repr, l_rec2_drop], axis=1,
                               name=id_tag + 'concat_context_rec2')
        l_hidden_drop = l_concat
        for i in range(1, self.options.listener_hidden_color_layers + 1):
            l_hidden = NINLayer(l_hidden_drop, num_units=self.options.listener_cell_size,
                                nonlinearity=NONLINEARITIES[self.options.listener_nonlinearity],
                                name=id_tag + 'hidden_combined%d' % i)
            if self.options.listener_dropout > 0.0:
                l_hidden_drop = DropoutLayer(l_hidden, p=self.options.listener_dropout,
                                             name=id_tag + 'hidden_drop')
            else:
                l_hidden_drop = l_hidden

        l_scores = DenseLayer(l_hidden_drop, num_units=self.context_len, nonlinearity=softmax,
                              name=id_tag + 'scores')

        return l_scores, [l_in] + context_inputs


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
        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_desc, get_color = (get_o, get_i) if inverted else (get_i, get_o)

        if init_vectorizer:
            self.seq_vec.add_all(get_desc(inst) for inst in training_instances)

        sentences = []
        colors = []
        if self.options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        for i, inst in enumerate(training_instances):
            self.word_counts.update([get_desc(inst)])
            desc = get_desc(inst)
            color = get_color(inst)
            if not color:
                assert test
                color = (0.0, 0.0, 0.0)
            if self.options.verbosity >= 9:
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
        id_tag = (self.id + '/') if self.id else ''
        input_var = T.ivector(id_tag + 'inputs')
        target_var = T.ivector(id_tag + 'targets')

        self.l_out, self.input_layers = self._get_l_out([input_var])
        self.loss = categorical_crossentropy

        self.model = model_class([input_var], [target_var], self.l_out,
                                 loss=self.loss, optimizer=rmsprop, id=self.id)

    def train_priors(self, training_instances, listener_data=False):
        prior_class = PRIORS[self.options.listener_prior]
        self.prior_emp = prior_class()  # TODO: accurate values for the empirical prior
        self.prior_smooth = prior_class()

        self.prior_emp.train(training_instances, listener_data=listener_data)
        self.prior_smooth.train(training_instances, listener_data=listener_data)

    def _get_l_out(self, input_vars):
        id_tag = (self.id + '/') if self.id else ''

        input_var = input_vars[0]

        l_in = InputLayer(shape=(None,), input_var=input_var,
                          name=id_tag + 'desc_input')
        embed_size = self.options.listener_cell_size or self.color_vec.num_types
        l_in_embed = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                                    output_size=embed_size,
                                    name=id_tag + 'desc_embed')

        if self.options.listener_cell_size == 0:
            l_scores = l_in_embed  # BiasLayer(l_in_embed, name=id_tag + 'bias')
        else:
            l_hidden = DenseLayer(l_in_embed, num_units=self.options.listener_cell_size,
                                  nonlinearity=NONLINEARITIES[self.options.listener_nonlinearity],
                                  name=id_tag + 'hidden')
            if self.options.listener_dropout > 0.0:
                l_hidden_drop = DropoutLayer(l_hidden, p=self.options.listener_dropout,
                                             name=id_tag + 'hidden_drop')
            else:
                l_hidden_drop = l_hidden

            l_scores = DenseLayer(l_hidden_drop, num_units=self.color_vec.num_types,
                                  nonlinearity=None, name=id_tag + 'scores')
        l_out = NonlinearityLayer(l_scores, nonlinearity=softmax, name=id_tag + 'out')

        return l_out, [l_in]

    def sample_prior_smooth(self, num_samples):
        return self.prior_smooth.sample(num_samples)


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
    'ContextListener': ContextListenerLearner,
    'TwoStreamListener': TwoStreamListenerLearner,
    'AtomicListener': AtomicListenerLearner,
}
