import numbers
import numpy as np
import theano.tensor as T
import warnings
from theano.tensor.nnet import crossentropy_categorical_1hot
from lasagne.layers import InputLayer, DropoutLayer, EmbeddingLayer, NonlinearityLayer, NINLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, DenseLayer, get_output, dimshuffle
from lasagne.layers.recurrent import Gate
from lasagne.init import Constant
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import rmsprop

from stanza.monitoring import progress
from stanza.research import config, iterators, instance
from stanza.research.rng import get_rng
import color_instances
from neural import NeuralLearner, SimpleLasagneModel
from neural import NONLINEARITIES, OPTIMIZERS, CELLS, sample
from vectorizers import SequenceVectorizer, SymbolVectorizer, strip_invalid_tokens, COLOR_REPRS
from vectorizers import BucketsVectorizer

parser = config.get_options_parser()
parser.add_argument('--speaker_cell_size', type=int, default=20,
                    help='The number of dimensions of all hidden layers and cells in '
                         'the speaker model. If 0 and using the AtomicSpeakerLearner, '
                         'remove all hidden layers and only train a linear classifier.')
parser.add_argument('--speaker_forget_bias', type=float, default=5.0,
                    help='The initial value of the forget gate bias in LSTM cells in '
                         'the speaker model. A positive initial forget gate bias '
                         'encourages the model to remember everything by default. '
                         'If speaker_cell is not LSTM, this value is ignored.')
parser.add_argument('--speaker_nonlinearity', choices=NONLINEARITIES.keys(), default='tanh',
                    help='The nonlinearity/activation function to use for dense and '
                         'recurrent layers in the speaker model.')
parser.add_argument('--speaker_cell', choices=CELLS.keys(), default='LSTM',
                    help='The recurrent cell to use for the speaker model.')
parser.add_argument('--speaker_dropout', type=float, default=0.2,
                    help='The dropout rate (probability of setting a value to zero). '
                         'Dropout will be disabled if nonpositive.')
parser.add_argument('--speaker_color_resolution', type=int, nargs='+', default=[4],
                    help='The number of buckets along each dimension of color space '
                         'for the input of the speaker model.')
parser.add_argument('--speaker_no_mask', type=config.boolean, default=False,
                    help='If `True`, disable masking of sequence inputs in training.')
parser.add_argument('--speaker_hidden_color_layers', type=int, default=0,
                    help='The number of dense layers after the color representation.')
parser.add_argument('--speaker_recurrent_layers', type=int, default=2,
                    help='The number of recurrent layers to pass the input through.')
parser.add_argument('--speaker_hidden_out_layers', type=int, default=0,
                    help='The number of dense layers to pass activations through '
                         'before the output.')
parser.add_argument('--speaker_hsv', type=config.boolean, default=False,
                    help='If `True`, input color buckets are in HSV space; otherwise, '
                         'color buckets will be in RGB. Input instances should be in HSV '
                         'regardless; this sets the internal representation for training '
                         'and prediction.')
parser.add_argument('--speaker_eval_batch_size', type=int, default=16384,
                    help='The number of examples per batch for evaluating the speaker '
                         'model. Higher means faster but more memory usage. This should '
                         'not affect modeling accuracy.')
parser.add_argument('--speaker_beam_size', type=int, default=1,
                    help='The number of choices to keep in memory at each time step '
                         'during prediction. Only used for recurrent speakers.')
parser.add_argument('--speaker_optimizer', choices=OPTIMIZERS.keys(), default='rmsprop',
                    help='The optimization (update) algorithm to use for speaker training.')
parser.add_argument('--speaker_learning_rate', type=float, default=0.1,
                    help='The learning rate to use for speaker training.')
parser.add_argument('--speaker_grad_clipping', type=float, default=0.0,
                    help='The maximum absolute value of the gradient messages for the'
                         'cell component of the speaker model.')
parser.add_argument('--speaker_color_repr', choices=COLOR_REPRS.keys(), default='buckets',
                    help='The representation of the color to use in the speaker model: a regular '
                         'grid of `buckets` or the `raw` RGB/HSV values.')

rng = get_rng()


class UniformPrior(object):
    '''A uniform color prior in RGB space.'''
    def __init__(self, recurrent=False):
        self.sampler = BucketsVectorizer([1], hsv=False)
        self.recurrent = recurrent

    def train(self, training_instances, listener_data='ignored'):
        pass

    def apply(self, input_vars):
        c = input_vars[0]
        if self.recurrent:
            if c.ndim == 2:
                ones = T.ones_like(c[:, 0])
            elif c.ndim == 3:
                ones = T.ones_like(c[:, 0, 0])
            else:
                assert False, 'need handling for higher rank color vectors (recurrent): %d' % c.ndim
        else:
            if c.ndim == 1:
                ones = T.ones_like(c)
            elif c.ndim == 2:
                ones = T.ones_like(c[:, 0])
            else:
                assert False, 'need handling for higher rank color vectors (atomic): %d' % c.ndim
        return -3.0 * np.log(256.0) * ones

    def sample(self, num_samples):
        '''
        :return: a list of `num_samples` colors sampled uniformly in RGB space,
                 but expressed as HSV triples.
        '''
        colors = self.sampler.unvectorize_all(np.zeros(num_samples, dtype=np.int32),
                                              random=True, hsv=True)
        return [instance.Instance(c) for c in colors]


class UniformContextPrior(UniformPrior):
    def __init__(self, recurrent=False):
        super(UniformContextPrior, self).__init__(recurrent=recurrent)

    def apply(self, input_vars):
        options = config.options()
        context_len = options.num_distractors
        return (super(UniformContextPrior, self).apply(input_vars) -
                3.0 * np.log(256.0) * context_len)

    def sample(self, num_samples=1):
        colors = super(UniformContextPrior, self).sample(num_samples)
        insts = [instance.Instance(c.input) for c in colors]
        return color_instances.reference_game(insts, color_instances.uniform, listener=False)


PRIORS = {
    'Uniform': UniformPrior,
    'UniformContext': UniformContextPrior,
}

parser.add_argument('--speaker_prior', choices=PRIORS.keys(), default='Uniform',
                    help='The prior model for the speaker (prior over colors). '
                         'Only used in RSA learner.')


class SpeakerLearner(NeuralLearner):
    '''
    An speaker with a feedforward neural net color input passed into an RNN
    to generate a description.
    '''
    def __init__(self, id=None, context_len=1):
        super(SpeakerLearner, self).__init__(id=id)
        self.seq_vec = SequenceVectorizer()
        color_repr = COLOR_REPRS[self.options.speaker_color_repr]
        self.color_vec = color_repr(self.options.speaker_color_resolution,
                                    hsv=self.options.speaker_hsv)
        self.context_len = context_len

    def predict(self, eval_instances, random=False, verbosity=0):
        result = []
        batches = iterators.iter_batches(eval_instances, self.options.speaker_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // self.options.speaker_eval_batch_size + 1

        eos_index = self.seq_vec.vectorize(['</s>'])[0]

        if self.options.verbosity + verbosity >= 2:
            print('Predicting')
        if self.options.verbosity + verbosity >= 1:
            progress.start_task('Predict batch', num_batches)
        for batch_num, batch in enumerate(batches):
            if self.options.verbosity + verbosity >= 1:
                progress.progress(batch_num)
            batch = list(batch)

            (c, _p, mask), (_y,) = self._data_to_arrays(batch, test=True)
            assert mask.all()  # We shouldn't be masking anything in prediction

            beam_size = 1 if random else self.options.speaker_beam_size
            done = np.zeros((len(batch), beam_size), dtype=np.bool)
            beam = np.zeros((len(batch), beam_size, self.seq_vec.max_len),
                            dtype=np.int32)
            beam[:, :, 0] = self.seq_vec.vectorize(['<s>'])[0]
            beam_scores = np.log(np.zeros((len(batch), beam_size)))
            beam_scores[:, 0] = 0.0

            c = np.repeat(c, beam_size, axis=0)
            mask = np.repeat(mask, beam_size, axis=0)

            for length in range(1, self.seq_vec.max_len):
                if done.all():
                    break
                p = beam.reshape((beam.shape[0] * beam.shape[1], beam.shape[2]))[:, :-1]
                probs = self.model.predict([c, p, mask])
                if random:
                    indices = sample(probs[:, length - 1, :])
                    beam[:, 0, length] = indices
                    done = np.logical_or(done, indices == eos_index)
                else:
                    assert probs.shape[1] == p.shape[1], (probs.shape[1], p.shape[1])
                    assert probs.shape[2] == len(self.seq_vec.tokens), (probs.shape[2],
                                                                        len(self.seq_vec.tokens))
                    scores = np.log(probs)[:, length - 1, :].reshape((beam.shape[0], beam.shape[1],
                                                                      probs.shape[2]))
                    beam_search_step(scores, length, beam, beam_scores, done, eos_index)
            outputs = self.seq_vec.unvectorize_all(beam[:, 0, :])
            result.extend([' '.join(strip_invalid_tokens(o)) for o in outputs])
        if self.options.verbosity + verbosity >= 1:
            progress.end_task()

        return result

    def score(self, eval_instances, verbosity=0):
        result = []
        batches = iterators.iter_batches(eval_instances, self.options.speaker_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // self.options.speaker_eval_batch_size + 1

        if self.options.verbosity + verbosity >= 2:
            print('Scoring')
        if self.options.verbosity + verbosity >= 1:
            progress.start_task('Score batch', num_batches)
        for batch_num, batch in enumerate(batches):
            if self.options.verbosity + verbosity >= 1:
                progress.progress(batch_num)
            batch = list(batch)

            xs, (n,) = self._data_to_arrays(batch, test=False)
            _, _, mask = xs

            probs = self.model.predict(xs)
            token_probs = probs[np.arange(probs.shape[0])[:, np.newaxis],
                                np.arange(probs.shape[1]), n]
            scores_arr = np.sum(np.log(token_probs) * mask, axis=1)
            scores = scores_arr.tolist()
            result.extend(scores)
        if self.options.verbosity + verbosity >= 1:
            progress.end_task()

        return result

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        context_len = self.context_len if hasattr(self, 'context_len') else 1
        use_context = context_len > 1

        def get_multi(val):
            if isinstance(val, tuple):
                assert len(val) == 1
                return val[0]
            else:
                return val

        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_color, get_desc_simple = (get_o, get_i) if inverted else (get_i, get_o)
        get_desc = lambda inst: get_multi(get_desc_simple(inst))
        get_i_ind, get_o_ind = ((lambda inst: inst.alt_inputs[get_multi(inst.input)]),
                                (lambda inst: inst.alt_outputs[get_multi(inst.output)]))
        get_color_indexed = get_o_ind if inverted else get_i_ind
        get_alt_i, get_alt_o = (lambda inst: inst.alt_inputs), (lambda inst: inst.alt_outputs)
        get_alt_colors = get_alt_o if inverted else get_alt_i

        if init_vectorizer:
            self.seq_vec.add_all(['<s>'] + get_desc(inst).split() + ['</s>']
                                 for inst in training_instances)

        colors = []
        previous = []
        next_tokens = []
        if self.options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        for i, inst in enumerate(training_instances):
            desc, color = get_desc(inst), get_color(inst)
            if isinstance(color, numbers.Number):
                color = get_color_indexed(inst)
            if test:
                full = ['<s>'] + ['</s>'] * (self.seq_vec.max_len - 1)
            else:
                desc = desc.split()
                full = (['<s>'] + desc + ['</s>'] +
                        ['<MASK>'] * (self.seq_vec.max_len - 1 - len(desc)))
            prev = full[:-1]
            next = full[1:]
            if self.options.verbosity >= 9:
                print('%s, %s -> %s' % (repr(color), repr(prev), repr(next)))
            colors.append(color)
            if use_context:
                new_context = get_alt_colors(inst)
                index = get_color(inst)
                if isinstance(index, tuple):
                    assert len(index) == 1
                    index = index[0]
                assert len(new_context) == context_len, \
                    'Inconsistent context lengths: %s' % ((context_len, len(new_context)),)
                colors.extend([c for j, c in enumerate(new_context) if j != index])
            previous.append(prev)
            next_tokens.append(next)

        P = np.zeros((len(previous), self.seq_vec.max_len - 1), dtype=np.int32)
        mask = np.zeros((len(previous), self.seq_vec.max_len - 1), dtype=np.int32)
        N = np.zeros((len(next_tokens), self.seq_vec.max_len - 1), dtype=np.int32)
        c = self.color_vec.vectorize_all(colors, hsv=True)
        if len(c.shape) == 1:
            c = c.reshape((len(colors) / context_len, context_len))
        else:
            c = c.reshape((len(colors) / context_len, context_len * c.shape[1]) +
                          c.shape[2:])
        for i, (color, prev, next) in enumerate(zip(colors, previous, next_tokens)):
            if len(prev) > P.shape[1]:
                prev = prev[:P.shape[1]]
            if len(next) > N.shape[1]:
                next = next[:N.shape[1]]
            P[i, :len(prev)] = self.seq_vec.vectorize(prev)
            N[i, :len(next)] = self.seq_vec.vectorize(next)
            for t, token in enumerate(next):
                mask[i, t] = (token != '<MASK>')
        c = np.tile(c[:, np.newaxis, ...], [1, self.seq_vec.max_len - 1] + [1] * (c.ndim - 1))

        if self.options.verbosity >= 9:
            print('c: %s' % (repr(c),))
            print('P: %s' % (repr(P),))
            print('mask: %s' % (repr(mask),))
            print('N: %s' % (repr(N),))
        return [c, P, mask], [N]

    def _build_model(self, model_class=SimpleLasagneModel):
        id_tag = (self.id + '/') if self.id else ''

        input_vars = self.color_vec.get_input_vars(self.id, recurrent=True) + [
            T.imatrix(id_tag + 'previous'),
            T.imatrix(id_tag + 'mask')
        ]
        target_var = T.imatrix(id_tag + 'targets')

        self.l_out, self.input_layers = self._get_l_out(input_vars)
        self.model = model_class(input_vars, [target_var], self.l_out, id=self.id,
                                 loss=self.masked_loss(input_vars),
                                 optimizer=OPTIMIZERS[self.options.speaker_optimizer],
                                 learning_rate=self.options.speaker_learning_rate)

    def train_priors(self, training_instances, listener_data=False):
        prior_class = PRIORS[self.options.speaker_prior]
        self.prior_emp = prior_class(recurrent=True)
        self.prior_smooth = prior_class(recurrent=True)

        self.prior_emp.train(training_instances, listener_data=listener_data)
        self.prior_smooth.train(training_instances, listener_data=listener_data)

    def _get_l_out(self, input_vars):
        check_options(self.options)
        id_tag = (self.id + '/') if self.id else ''

        prev_output_var, mask_var = input_vars[-2:]
        color_input_vars = input_vars[:-2]

        context_len = self.context_len if hasattr(self, 'context_len') else 1
        l_color_repr, color_inputs = self.color_vec.get_input_layer(
            color_input_vars,
            recurrent_length=self.seq_vec.max_len - 1,
            cell_size=self.options.speaker_cell_size,
            context_len=context_len,
            id=self.id
        )
        l_hidden_color = dimshuffle(l_color_repr, (0, 2, 1))
        for i in range(1, self.options.speaker_hidden_color_layers + 1):
            l_hidden_color = NINLayer(
                l_hidden_color, num_units=self.options.speaker_cell_size,
                nonlinearity=NONLINEARITIES[self.options.speaker_nonlinearity],
                name=id_tag + 'hidden_color%d' % i)
        l_hidden_color = dimshuffle(l_hidden_color, (0, 2, 1))

        l_prev_out = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                                input_var=prev_output_var,
                                name=id_tag + 'prev_input')
        l_prev_embed = EmbeddingLayer(l_prev_out, input_size=len(self.seq_vec.tokens),
                                      output_size=self.options.speaker_cell_size,
                                      name=id_tag + 'prev_embed')
        l_in = ConcatLayer([l_hidden_color, l_prev_embed], axis=2, name=id_tag + 'color_prev')
        l_mask_in = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                               input_var=mask_var, name=id_tag + 'mask_input')
        l_rec_drop = l_in

        cell = CELLS[self.options.speaker_cell]
        cell_kwargs = {
            'mask_input': (None if self.options.speaker_no_mask else l_mask_in),
            'grad_clipping': self.options.speaker_grad_clipping,
            'num_units': self.options.speaker_cell_size,
        }
        if self.options.speaker_cell == 'LSTM':
            cell_kwargs['forgetgate'] = Gate(b=Constant(self.options.speaker_forget_bias))
        if self.options.speaker_cell != 'GRU':
            cell_kwargs['nonlinearity'] = NONLINEARITIES[self.options.speaker_nonlinearity]

        for i in range(1, self.options.speaker_recurrent_layers):
            l_rec = cell(l_rec_drop, name=id_tag + 'rec%d' % i, **cell_kwargs)
            if self.options.speaker_dropout > 0.0:
                l_rec_drop = DropoutLayer(l_rec, p=self.options.speaker_dropout,
                                          name=id_tag + 'rec%d_drop' % i)
            else:
                l_rec_drop = l_rec
        l_rec = cell(l_rec_drop, name=id_tag + 'rec%d' % self.options.speaker_recurrent_layers,
                     **cell_kwargs)
        l_shape = ReshapeLayer(l_rec, (-1, self.options.speaker_cell_size),
                               name=id_tag + 'reshape')
        l_hidden_out = l_shape
        for i in range(1, self.options.speaker_hidden_out_layers + 1):
            l_hidden_out = DenseLayer(
                l_hidden_out, num_units=self.options.speaker_cell_size,
                nonlinearity=NONLINEARITIES[self.options.speaker_nonlinearity],
                name=id_tag + 'hidden_out%d' % i)
        l_softmax = DenseLayer(l_hidden_out, num_units=len(self.seq_vec.tokens),
                               nonlinearity=softmax, name=id_tag + 'softmax')
        l_out = ReshapeLayer(l_softmax, (-1, self.seq_vec.max_len - 1, len(self.seq_vec.tokens)),
                             name=id_tag + 'out')

        return l_out, color_inputs + [l_prev_out, l_mask_in]

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

    def sample_prior_smooth(self, num_samples):
        return self.prior_smooth.sample(num_samples)


class ContextSpeakerLearner(SpeakerLearner):
    def __init__(self, *args, **kwargs):
        self.get_options()
        context = self.options.num_distractors + 1
        return super(ContextSpeakerLearner, self).__init__(*args, context_len=context, **kwargs)


def check_options(options):
    if options.speaker_grad_clipping:
        warnings.warn('Per-dimension gradient clipping (--speaker_grad_clipping) is enabled. '
                      'This feature is unlikely to correctly constrain gradients and avoid '
                      'NaNs; use --true_grad_clipping instead.')
    if options.speaker_recurrent_layers and not options.true_grad_clipping:
        warnings.warn('Norm-constraint gradient clipping is disabled for a recurrent model. '
                      'This will likely lead to exploding gradients.')
    if options.speaker_recurrent_layers and options.true_grad_clipping > 6.0:
        warnings.warn('Gradient clipping norm is unusually high (%s). '
                      'This could lead to exploding gradients.' % options.true_grad_clipping)
    if options.speaker_nonlinearity == 'rectify':
        warnings.warn('Using ReLU as the output nonlinearity for a recurrent unit. This may '
                      'be a source of NaNs in the gradient.')


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


def beam_search_step(scores, length, beam, beam_scores, done, eos_index):
    '''
    Perform one step of beam search, given the matrix of probabilities
    for each possible following token.

    Modifies `beam`, `beam_scores`, and `done` *in place*.

    :param scores: Scores (log probabilities, up to a constant) assigned by the
        model to each token for each sequence on the various beams.
    :type scores: float ndarray, shape `(batch_size, beam_size, vocab_size)`
    :param int length: Current length of already predicted sequences.
        Should equal the axis-1 index in `beam` where the next
        predicted tokens will be populated.
    :param beam: Token indices for the top-k sequences predicted for each
        example.
    :type beam: int ndarray, shape `(batch_size, beam_size, max_seq_len)`
    :param beam_scores: log probabilities assigned to current candidate sequences
    :type beam_scores: float ndarray, shape `(batch_size, beam_size)`
    :param done: Mask of beam entries that have reached the &lt;/s&gt; token
    :type done: boolean ndarray, shape `(batch_size, beam_size)`

    As an example, suppose the distribution represented by the model is:

        'a cat': 0.375,
        'cat': 0.25,
        'cat a': 0.125,
        'cat cat': 0.125,
        'a': 0.0625,
        '': 0.03125,
        'a a': 0.03125,

    >>> a_cat,   cat, cat_a, cat_cat,    a,    null,     a_a = \\
    ... [0.375, 0.25, 0.125, 0.125, 0.0625, 0.03125, 0.03125]

    >>> vec = SequenceVectorizer(); vec.add(['<s>', 'a', 'cat', '</s>'])
    >>> vec.vectorize(['<s>', 'a', 'cat', '</s>'])
    array([1, 2, 3, 4], dtype=int32)
    >>> eos_index = vec.vectorize(['</s>'])[0]

    Initialize the beam. Note that -inf should be the initial score
    for all but one item on each beam; if all scores start at 0,
    the beam will be saturated with duplicates of the greedy choice.

    >>> batch_size = 1; beam_size = 2; max_seq_len = 3
    >>> beam = np.zeros((batch_size, beam_size, max_seq_len), dtype=np.int)
    >>> beam_scores = np.log(np.zeros((batch_size, beam_size)))
    >>> beam_scores[:, 0] = 0.0
    >>> done = np.zeros((batch_size, beam_size), dtype=np.bool)

    >>> next_scores = np.log([[[0.0, 0.0,
    ...                         a_cat + a + a_a,
    ...                         cat + cat_cat + cat_a,
    ...                         null]] * 2])
    >>> beam_search_step(next_scores, 0, beam, beam_scores, done, eos_index)
    >>> beam
    array([[[3, 0, 0],
            [2, 0, 0]]])
    >>> np.exp(beam_scores).round(5)
    array([[ 0.5    ,  0.46875]])
    >>> done
    array([[False, False]], dtype=bool)

    Note that 'cat' is the greedy first choice, but 'a cat' will end up
    with a higher score.

    >>> next_scores = np.log([[[0.0, 0.0, cat_a / 0.5, cat_cat / 0.5, cat / 0.5],
    ...                        [0.0, 0.0, a_a / 0.46875, a_cat / 0.46875, a / 0.46875]]])
    >>> beam_search_step(next_scores, 1, beam, beam_scores, done, eos_index)
    >>> beam
    array([[[2, 3, 0],
            [3, 4, 0]]])
    >>> np.exp(beam_scores).round(3)
    array([[ 0.375,  0.25 ]])
    >>> done
    array([[False,  True]], dtype=bool)

    The best sequences have been identified; the score for 'cat' stays constant
    after it reaches the end-of-sentence token, and the beam is padded with
    end-of-sentence tokens regardless of the returned scores.

    >>> next_scores = np.log([[[0.0, 0.0, 0.0, 0.0, 1.0],
    ...                        [0.0, 0.0, 0.25, 0.5, 0.25]]])
    >>> beam_search_step(next_scores, 2, beam, beam_scores, done, eos_index)
    >>> beam
    array([[[2, 3, 4],
            [3, 4, 4]]])
    >>> np.exp(beam_scores).round(3)
    array([[ 0.375,  0.25 ]])
    >>> done
    array([[ True,  True]], dtype=bool)
    '''
    assert len(scores.shape) == 3, scores.shape
    batch_size, beam_size, vocab_size = scores.shape
    assert len(beam.shape) == 3, beam.shape
    assert beam.shape[:2] == (batch_size, beam_size), \
        '%s != (%s, %s, *)' % (beam.shape, batch_size, beam_size)
    max_seq_len = beam.shape[2]
    assert beam_scores.shape == (batch_size, beam_size), \
        '%s != %s' % (beam_scores.shape, (batch_size, beam_size))
    assert done.shape == (batch_size, beam_size), \
        '%s != %s' % (done.shape, (batch_size, beam_size))

    # Compute updated scores
    new_scores = (scores * ~done[:, :, np.newaxis] +
                  beam_scores[:, :, np.newaxis]).reshape((batch_size, beam_size * vocab_size))
    # Get indices of top k scores
    topk = np.argsort(-new_scores)[:, :beam_size]
    # Transform into previous beam indices and new token indices
    rows, new_indices = np.unravel_index(topk, (beam_size, vocab_size))
    assert rows.shape == (batch_size, beam_size), \
        '%s != %s' % (rows.shape, (batch_size, beam_size))
    assert new_indices.shape == (batch_size, beam_size), \
        '%s != %s' % (new_indices.shape, (batch_size, beam_size))

    # Extract best pre-existing rows
    beam[:, :, :] = beam[np.arange(batch_size)[:, np.newaxis], rows, :]
    assert beam.shape == (batch_size, beam_size, max_seq_len), \
        '%s != %s' % (beam.shape, (batch_size, beam_size, max_seq_len))
    # Append new token indices
    beam[:, :, length] = new_indices
    # Update beam scores
    beam_scores[:, :] = new_scores[np.arange(batch_size)[:, np.newaxis], topk]
    # Get previous done status and update it with
    # which rows have newly reached </s>
    done[:, :] = done[np.arange(batch_size)[:, np.newaxis], rows] | (new_indices == eos_index)
    # Pad already-finished sequences with </s>
    beam[done, length] = eos_index


class AtomicSpeakerLearner(NeuralLearner):
    '''
    A speaker that learns to produce descriptions as indivisible symbols (as
    opposed to word-by-word sequences) given colors.
    '''
    def __init__(self, id=None):
        super(AtomicSpeakerLearner, self).__init__(id=id)
        self.seq_vec = SymbolVectorizer()
        color_repr = COLOR_REPRS[self.options.speaker_color_repr]
        self.color_vec = color_repr(self.options.speaker_color_resolution,
                                    hsv=self.options.speaker_hsv)

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        predictions = []
        scores = []
        batches = iterators.iter_batches(eval_instances, self.options.speaker_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // self.options.speaker_eval_batch_size + 1

        if self.options.verbosity + verbosity >= 2:
            print('Testing')
        if self.options.verbosity + verbosity >= 1:
            progress.start_task('Eval batch', num_batches)
        for batch_num, batch in enumerate(batches):
            if self.options.verbosity + verbosity >= 1:
                progress.progress(batch_num)
            batch = list(batch)

            xs, (y,) = self._data_to_arrays(batch, test=True)

            probs = self.model.predict(xs)
            if random:
                indices = sample(probs)
            else:
                indices = probs.argmax(axis=1)
            predictions.extend(self.seq_vec.unvectorize_all(indices))
            scores_arr = np.log(probs[np.arange(len(batch)), y])
            scores.extend(scores_arr.tolist())
        if self.options.verbosity + verbosity >= 1:
            progress.end_task()
        if self.options.verbosity >= 9:
            print('%s %ss:') % (self.id, 'sample' if random else 'prediction')
            for inst, prediction in zip(eval_instances, predictions):
                print('%s -> %s' % (repr(inst.input), repr(prediction)))

        return predictions, scores

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_color, get_desc = (get_o, get_i) if inverted else (get_i, get_o)

        if init_vectorizer:
            self.seq_vec.add_all(get_desc(inst) for inst in training_instances)

        sentences = []
        colors = []
        if self.options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        for i, inst in enumerate(training_instances):
            desc = get_desc(inst)
            if desc is None:
                assert test
                desc = '<unk>'
            color = get_color(inst)
            assert color
            if self.options.verbosity >= 9:
                print('%s -> %s' % (repr(desc), repr(color)))
            sentences.append(desc)
            colors.append(color)

        x = self.color_vec.vectorize_all(colors, hsv=True)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        y = self.seq_vec.vectorize_all(sentences)
        if self.options.verbosity >= 9:
            print('%s x: %s' % (self.id, x))
            print('%s y: %s' % (self.id, y))

        return [x], [y]

    def _build_model(self, model_class=SimpleLasagneModel):
        id_tag = (self.id + '/') if self.id else ''
        input_vars = self.color_vec.get_input_vars(self.id)
        target_var = T.ivector(id_tag + 'targets')

        self.l_out, self.input_layers = self._get_l_out(input_vars)
        self.loss = categorical_crossentropy

        self.model = model_class(input_vars, [target_var], self.l_out,
                                 loss=self.loss, optimizer=rmsprop, id=self.id)

    def train_priors(self, training_instances, listener_data=False):
        prior_class = PRIORS[self.options.speaker_prior]
        self.prior_emp = prior_class()
        self.prior_smooth = prior_class()

        self.prior_emp.train(training_instances, listener_data=listener_data)
        self.prior_smooth.train(training_instances, listener_data=listener_data)

    def _get_l_out(self, input_vars):
        id_tag = (self.id + '/') if self.id else ''

        cell_size = self.options.speaker_cell_size or self.seq_vec.num_types
        l_color_repr, color_inputs = self.color_vec.get_input_layer(
            input_vars,
            recurrent_length=0,
            cell_size=cell_size,
            id=self.id
        )
        l_hidden_color = l_color_repr
        for i in range(1, self.options.speaker_hidden_color_layers + 1):
            l_hidden_color = NINLayer(
                l_hidden_color, num_units=cell_size,
                nonlinearity=NONLINEARITIES[self.options.speaker_nonlinearity],
                name=id_tag + 'hidden_color%d' % i)
        l_hidden_color = l_hidden_color

        if self.options.speaker_cell_size == 0:
            l_scores = l_color_repr  # BiasLayer(l_color_repr, name=id_tag + 'bias')
        else:
            if self.options.speaker_dropout > 0.0:
                l_color_drop = DropoutLayer(l_hidden_color, p=self.options.speaker_dropout,
                                            name=id_tag + 'color_drop')
            else:
                l_color_drop = l_hidden_color

            l_hidden = DenseLayer(l_color_drop, num_units=self.options.speaker_cell_size,
                                  nonlinearity=NONLINEARITIES[self.options.speaker_nonlinearity],
                                  name=id_tag + 'hidden')
            if self.options.speaker_dropout > 0.0:
                l_hidden_drop = DropoutLayer(l_hidden, p=self.options.speaker_dropout,
                                             name=id_tag + 'hidden_drop')
            else:
                l_hidden_drop = l_hidden

            l_scores = DenseLayer(l_hidden_drop, num_units=self.seq_vec.num_types,
                                  nonlinearity=None, name=id_tag + 'scores')
        l_out = NonlinearityLayer(l_scores, nonlinearity=softmax,
                                  name=id_tag + 'softmax')

        return l_out, color_inputs

    def sample_prior_smooth(self, num_samples):
        return self.prior_smooth.sample(num_samples)


SPEAKERS = {
    'Speaker': SpeakerLearner,
    'ContextSpeaker': ContextSpeakerLearner,
    'AtomicSpeaker': AtomicSpeakerLearner,
}
