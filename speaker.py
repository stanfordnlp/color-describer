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

from stanza.unstable import config, progress, iterators, instance
from stanza.unstable.rng import get_rng
from neural import NeuralLearner, SimpleLasagneModel
from neural import NONLINEARITIES, OPTIMIZERS, CELLS, sample
from vectorizers import SequenceVectorizer, SymbolVectorizer
from vectorizers import BucketsVectorizer, RawVectorizer, MSVectorizer, FourierVectorizer

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
parser.add_argument('--speaker_no_mask', action='store_true',
                    help='If `True`, disable masking of sequence inputs in training.')
parser.add_argument('--speaker_hidden_color_layers', type=int, default=0,
                    help='The number of dense layers after the color representation.')
parser.add_argument('--speaker_recurrent_layers', type=int, default=2,
                    help='The number of recurrent layers to pass the input through.')
parser.add_argument('--speaker_hidden_out_layers', type=int, default=0,
                    help='The number of dense layers to pass activations through '
                         'before the output.')
parser.add_argument('--speaker_hsv', action='store_true',
                    help='If `True`, input color buckets are in HSV space; otherwise, '
                         'color buckets will be in RGB. Input instances should be in HSV '
                         'regardless; this sets the internal representation for training '
                         'and prediction.')
parser.add_argument('--speaker_eval_batch_size', type=int, default=16384,
                    help='The number of examples per batch for evaluating the speaker '
                         'model. Higher means faster but more memory usage. This should '
                         'not affect modeling accuracy.')
parser.add_argument('--speaker_optimizer', choices=OPTIMIZERS.keys(), default='rmsprop',
                    help='The optimization (update) algorithm to use for speaker training.')
parser.add_argument('--speaker_learning_rate', type=float, default=0.1,
                    help='The learning rate to use for speaker training.')
parser.add_argument('--speaker_grad_clipping', type=float, default=0.0,
                    help='The maximum absolute value of the gradient messages for the'
                         'cell component of the speaker model.')


COLOR_REPRS = {
    'raw': RawVectorizer,
    'buckets': BucketsVectorizer,
    'ms': MSVectorizer,
    'fourier': FourierVectorizer,
}

parser.add_argument('--speaker_color_repr', choices=COLOR_REPRS.keys(), default='buckets',
                    help='The representation of the color to use in the speaker model: a regular '
                         'grid of `buckets` or the `raw` RGB/HSV values.')

rng = get_rng()


class UniformPrior(object):
    '''A uniform color prior in RGB space.'''
    def __init__(self):
        self.sampler = BucketsVectorizer([1], hsv=False)

    def fit(self, xs, ys):
        pass

    def apply(self, input_vars):
        c = input_vars[0]
        if c.ndim == 1:
            ones = T.ones_like(c)
        else:
            ones = T.ones_like(c[:, 0])
        return -3.0 * np.log(256.0) * ones

    def sample(self, num_samples):
        '''
        :return: a list of `num_samples` colors sampled uniformly in RGB space,
                 but expressed as HSV triples.
        '''
        return self.sampler.unvectorize_all(np.zeros(num_samples, dtype=np.int32),
                                            random=True, hsv=True)


class SpeakerLearner(NeuralLearner):
    '''
    An speaker with a feedforward neural net color input passed into an RNN
    to generate a description.
    '''
    def __init__(self, id=None):
        super(SpeakerLearner, self).__init__(id=id)
        options = config.options()
        self.seq_vec = SequenceVectorizer()
        color_repr = COLOR_REPRS[options.speaker_color_repr]
        self.color_vec = color_repr(options.speaker_color_resolution,
                                    hsv=options.speaker_hsv)

    def predict(self, eval_instances, random=False, verbosity=0):
        options = config.options()

        result = []
        batches = iterators.iter_batches(eval_instances, options.speaker_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // options.speaker_eval_batch_size + 1

        if options.verbosity + verbosity >= 2:
            print('Predicting')
        progress.start_task('Predict batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)

            (c, _p, mask), (_y,) = self._data_to_arrays(batch, test=True)

            done = np.zeros((len(batch),), dtype=np.bool)
            outputs = [['<s>'] + ['</s>'] * (self.seq_vec.max_len - 2)
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

    def score(self, eval_instances, verbosity=0):
        options = config.options()

        result = []
        batches = iterators.iter_batches(eval_instances, options.speaker_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // options.speaker_eval_batch_size + 1

        if options.verbosity + verbosity >= 2:
            print('Scoring')
        progress.start_task('Score batch', num_batches)
        for batch_num, batch in enumerate(batches):
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
        progress.end_task()

        return result

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        options = config.options()

        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_color, get_desc = (get_o, get_i) if inverted else (get_i, get_o)

        if init_vectorizer:
            self.seq_vec.add_all(['<s>'] + get_desc(inst).split() + ['</s>']
                                 for inst in training_instances)

        colors = []
        previous = []
        next_tokens = []
        if options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        for i, inst in enumerate(training_instances):
            desc, color = get_desc(inst), get_color(inst)
            if test:
                full = ['<s>'] + ['</s>'] * (self.seq_vec.max_len - 2)
            else:
                desc = desc.split()
                full = (['<s>'] + desc + ['</s>'] +
                        ['<MASK>'] * (self.seq_vec.max_len - 2 - len(desc)))
            prev = full[:-1]
            next = full[1:]
            if options.verbosity >= 9:
                print('%s, %s -> %s' % (repr(color), repr(prev), repr(next)))
            colors.append(color)
            previous.append(prev)
            next_tokens.append(next)

        P = np.zeros((len(previous), self.seq_vec.max_len - 1), dtype=np.int32)
        mask = np.zeros((len(previous), self.seq_vec.max_len - 1), dtype=np.int32)
        N = np.zeros((len(next_tokens), self.seq_vec.max_len - 1), dtype=np.int32)
        c = self.color_vec.vectorize_all(colors, hsv=True)
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

        if options.verbosity >= 9:
            print('c: %s' % (repr(c),))
            print('P: %s' % (repr(P),))
            print('mask: %s' % (repr(mask),))
            print('N: %s' % (repr(N),))
        return [c, P, mask], [N]

    def _build_model(self, model_class=SimpleLasagneModel):
        options = config.options()
        id_tag = (self.id + '/') if self.id else ''

        input_vars = self.color_vec.get_input_vars(self.id, recurrent=True) + [
            T.imatrix(id_tag + 'previous'),
            T.imatrix(id_tag + 'mask')
        ]
        target_var = T.imatrix(id_tag + 'targets')

        self.l_out, self.input_layers = self. _get_l_out(input_vars)
        self.model = model_class(input_vars, [target_var], self.l_out, id=self.id,
                                 loss=self.masked_loss(input_vars),
                                 optimizer=OPTIMIZERS[options.speaker_optimizer],
                                 learning_rate=options.speaker_learning_rate)

        self.prior_emp = UniformPrior()
        self.prior_smooth = UniformPrior()

    def _get_l_out(self, input_vars):
        options = config.options()
        check_options(options)
        id_tag = (self.id + '/') if self.id else ''

        prev_output_var, mask_var = input_vars[-2:]
        color_input_vars = input_vars[:-2]

        l_color_repr, color_inputs = self.color_vec.get_input_layer(
            color_input_vars,
            recurrent_length=self.seq_vec.max_len - 1,
            cell_size=options.speaker_cell_size,
            id=self.id
        )
        l_hidden_color = dimshuffle(l_color_repr, (0, 2, 1))
        for i in range(1, options.speaker_hidden_color_layers + 1):
            l_hidden_color = NINLayer(l_hidden_color, num_units=options.speaker_cell_size,
                                      nonlinearity=NONLINEARITIES[options.speaker_nonlinearity],
                                      name=id_tag + 'hidden_color%d' % i)
        l_hidden_color = dimshuffle(l_hidden_color, (0, 2, 1))

        l_prev_out = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                                input_var=prev_output_var,
                                name=id_tag + 'prev_input')
        l_prev_embed = EmbeddingLayer(l_prev_out, input_size=len(self.seq_vec.tokens),
                                      output_size=options.speaker_cell_size,
                                      name=id_tag + 'prev_embed')
        l_in = ConcatLayer([l_hidden_color, l_prev_embed], axis=2, name=id_tag + 'color_prev')
        l_mask_in = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                               input_var=mask_var, name=id_tag + 'mask_input')
        l_rec_drop = l_in

        cell = CELLS[options.speaker_cell]
        cell_kwargs = {
            'mask_input': (None if options.speaker_no_mask else l_mask_in),
            'grad_clipping': options.speaker_grad_clipping,
            'num_units': options.speaker_cell_size,
        }
        if options.speaker_cell == 'LSTM':
            cell_kwargs['forgetgate'] = Gate(b=Constant(options.speaker_forget_bias))
        if options.speaker_cell != 'GRU':
            cell_kwargs['nonlinearity'] = NONLINEARITIES[options.speaker_nonlinearity]

        for i in range(1, options.speaker_recurrent_layers):
            l_rec = cell(l_rec_drop, name=id_tag + 'rec%d' % i, **cell_kwargs)
            if options.speaker_dropout > 0.0:
                l_rec_drop = DropoutLayer(l_rec, p=options.speaker_dropout,
                                          name=id_tag + 'rec%d_drop' % i)
            else:
                l_rec_drop = l_rec
        l_rec = cell(l_rec_drop, name=id_tag + 'rec%d' % options.speaker_recurrent_layers,
                     **cell_kwargs)
        l_shape = ReshapeLayer(l_rec, (-1, options.speaker_cell_size),
                               name=id_tag + 'reshape')
        l_hidden_out = l_shape
        for i in range(1, options.speaker_hidden_out_layers + 1):
            l_hidden_out = DenseLayer(l_hidden_out, num_units=options.speaker_cell_size,
                                      nonlinearity=NONLINEARITIES[options.speaker_nonlinearity],
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
        return [instance.Instance(input=c) for c in
                self.prior_smooth.sample(num_samples)]


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


class AtomicSpeakerLearner(NeuralLearner):
    '''
    A speaker that learns to produce descriptions as indivisible symbols (as
    opposed to word-by-word sequences) given colors.
    '''
    def __init__(self, id=None):
        super(AtomicSpeakerLearner, self).__init__(id=id)
        options = config.options()
        self.seq_vec = SymbolVectorizer()
        color_repr = COLOR_REPRS[options.speaker_color_repr]
        self.color_vec = color_repr(options.speaker_color_resolution,
                                    hsv=options.speaker_hsv)

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        options = config.options()

        predictions = []
        scores = []
        batches = iterators.iter_batches(eval_instances, options.speaker_eval_batch_size)
        num_batches = (len(eval_instances) - 1) // options.speaker_eval_batch_size + 1

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
            else:
                indices = probs.argmax(axis=1)
            predictions.extend(self.seq_vec.unvectorize_all(indices))
            scores_arr = np.log(probs[np.arange(len(batch)), y])
            scores.extend(scores_arr.tolist())
        progress.end_task()
        if options.verbosity >= 9:
            print('%s %ss:') % (self.id, 'sample' if random else 'prediction')
            for inst, prediction in zip(eval_instances, predictions):
                print('%s -> %s' % (repr(inst.input), repr(prediction)))

        return predictions, scores

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        options = config.options()

        get_i, get_o = (lambda inst: inst.input), (lambda inst: inst.output)
        get_color, get_desc = (get_o, get_i) if inverted else (get_i, get_o)

        if init_vectorizer:
            self.seq_vec.add_all(get_desc(inst) for inst in training_instances)

        sentences = []
        colors = []
        if options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        for i, inst in enumerate(training_instances):
            desc = get_desc(inst)
            if desc is None:
                assert test
                desc = '<unk>'
            color = get_color(inst)
            assert color
            if options.verbosity >= 9:
                print('%s -> %s' % (repr(desc), repr(color)))
            sentences.append(desc)
            colors.append(color)

        x = self.color_vec.vectorize_all(colors, hsv=True)
        y = self.seq_vec.vectorize_all(sentences)
        if options.verbosity >= 9:
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

        self.prior_emp = UniformPrior()
        self.prior_smooth = UniformPrior()

    def _get_l_out(self, input_vars):
        options = config.options()
        id_tag = (self.id + '/') if self.id else ''

        cell_size = options.speaker_cell_size or self.seq_vec.num_types
        l_color_repr, color_inputs = self.color_vec.get_input_layer(
            input_vars,
            recurrent_length=0,
            cell_size=cell_size,
            id=self.id
        )
        l_hidden_color = l_color_repr
        for i in range(1, options.speaker_hidden_color_layers + 1):
            l_hidden_color = NINLayer(l_hidden_color, num_units=cell_size,
                                      nonlinearity=NONLINEARITIES[options.speaker_nonlinearity],
                                      name=id_tag + 'hidden_color%d' % i)
        l_hidden_color = l_hidden_color

        if options.speaker_cell_size == 0:
            l_scores = l_color_repr  # BiasLayer(l_color_repr, name=id_tag + 'bias')
        else:
            if options.speaker_dropout > 0.0:
                l_color_drop = DropoutLayer(l_hidden_color, p=options.speaker_dropout,
                                            name=id_tag + 'color_drop')
            else:
                l_color_drop = l_hidden_color

            l_hidden = DenseLayer(l_color_drop, num_units=options.speaker_cell_size,
                                  nonlinearity=NONLINEARITIES[options.speaker_nonlinearity],
                                  name=id_tag + 'hidden')
            if options.speaker_dropout > 0.0:
                l_hidden_drop = DropoutLayer(l_hidden, p=options.speaker_dropout,
                                             name=id_tag + 'hidden_drop')
            else:
                l_hidden_drop = l_hidden

            l_scores = DenseLayer(l_hidden_drop, num_units=self.seq_vec.num_types,
                                  nonlinearity=None, name=id_tag + 'scores')
        l_out = NonlinearityLayer(l_scores, nonlinearity=softmax,
                                  name=id_tag + 'softmax')

        return l_out, color_inputs

    def sample_prior_smooth(self, num_samples):
        return [instance.Instance(input=c) for c in
                self.prior_smooth.sample(num_samples)]


SPEAKERS = {
    'Speaker': SpeakerLearner,
    'AtomicSpeaker': AtomicSpeakerLearner,
}
