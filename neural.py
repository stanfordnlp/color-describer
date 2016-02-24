import colorsys
import lasagne
import numpy as np
import operator
import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as G
import time
from collections import Sequence, OrderedDict
from lasagne.layers import get_output, get_all_params
from lasagne.updates import total_norm_constraint
from matplotlib.colors import hsv_to_rgb
from theano.compile import MonitorMode
from theano.printing import pydotprint

from helpers import apply_nan_suppression
from stanza.unstable import config, progress, summary
from stanza.unstable.learner import Learner
from stanza.unstable.rng import get_rng

parser = config.get_options_parser()
parser.add_argument('--train_iters', type=int, default=10,
                    help='Number of iterations')
parser.add_argument('--train_epochs', type=int, default=100,
                    help='Number of epochs per iteration')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of examples per minibatch for training and evaluation')
parser.add_argument('--detect_nans', action='store_true',
                    help='If True, throw an error if a non-finite value is detected.')
parser.add_argument('--verbosity', type=int, default=4,
                    help='Amount of diagnostic output to produce. 0-1: only progress updates; '
                         '2-3: plus major experiment steps; '
                         '4-5: plus compilation and graph assembly steps; '
                         '6-7: plus parameter names for each function compilation; '
                         '8: plus shapes and types for each compiled function call; '
                         '9-10: plus vectorization of all datasets')
parser.add_argument('--no_graphviz', action='store_true',
                    help='If `True`, do not use theano.printing.pydotprint to visualize '
                         'function graphs.')
parser.add_argument('--no_nan_suppression', action='store_true',
                    help='If `True`, do not try to suppress NaNs in training.')
parser.add_argument('--monitor_grads', action='store_true',
                    help='If `True`, return gradients for monitoring and write them to the '
                         'TensorBoard events file.')
parser.add_argument('--monitor_params', action='store_true',
                    help='If `True`, write parameter value histograms out to the '
                         'TensorBoard events file.')
parser.add_argument('--true_grad_clipping', type=float, default=0.0,
                    help='The maximum absolute value of all gradients. This gradient '
                         'clipping is performed on the full gradient calculation, not '
                         'just the messages passing through the LSTM.')


NONLINEARITIES = {
    name: func
    for name, func in lasagne.nonlinearities.__dict__.iteritems()
    if name.islower() and not name.startswith('__')
}
del NONLINEARITIES['theano']

OPTIMIZERS = {
    name: func
    for name, func in lasagne.updates.__dict__.iteritems()
    if (name in lasagne.updates.__all__ and
        not name.startswith('apply_') and not name.endswith('_constraint'))
}

CELLS = {
    name[:-len('Layer')]: func
    for name, func in lasagne.layers.recurrent.__dict__.iteritems()
    if (name in lasagne.layers.recurrent.__all__ and name.endswith('Layer') and
        name != 'CustomRecurrentLayer')
}

rng = get_rng()
lasagne.random.set_rng(rng)


def detect_nan(i, node, fn):
    if not isinstance(node.op, (T.AllocEmpty, T.IncSubtensor,
                                G.GpuAllocEmpty, G.GpuIncSubtensor)):
        for output in fn.outputs:
            if (not isinstance(output[0], np.random.RandomState) and
                    not np.isfinite(output[0]).all()):
                print('*** NaN detected ***')
                theano.printing.debugprint(node)
                print('Inputs : %s' % [input[0] for input in fn.inputs])
                print('Outputs: %s' % [output[0] for output in fn.outputs])
                raise AssertionError


class SymbolVectorizer(object):
    '''
    Maps symbols from an alphabet/vocabulary of indefinite size to and from
    sequential integer ids.

    >>> vec = SymbolVectorizer()
    >>> vec.add_all(['larry', 'moe', 'larry', 'curly', 'moe'])
    >>> vec.vectorize_all(['curly', 'larry', 'moe', 'pikachu'])
    array([3, 1, 2, 0], dtype=int32)
    >>> vec.unvectorize_all([3, 3, 2])
    ['curly', 'curly', 'moe']
    '''
    def __init__(self):
        self.tokens = []
        self.token_indices = {}
        self.indices_token = {}
        self.add('<unk>')

    @property
    def num_types(self):
        return len(self.tokens)

    def add_all(self, symbols):
        for sym in symbols:
            self.add(sym)

    def add(self, symbol):
        if symbol not in self.token_indices:
            self.token_indices[symbol] = len(self.tokens)
            self.indices_token[len(self.tokens)] = symbol
            self.tokens.append(symbol)

    def vectorize(self, symbol):
        return (self.token_indices[symbol] if symbol in self.token_indices
                else self.token_indices['<unk>'])

    def vectorize_all(self, symbols):
        return np.array([self.vectorize(sym) for sym in symbols], dtype=np.int32)

    def unvectorize(self, index):
        return self.indices_token[index]

    def unvectorize_all(self, array):
        if hasattr(array, 'tolist'):
            array = array.tolist()
        return [self.unvectorize(elem) for elem in array]


class SequenceVectorizer(object):
    '''
    Maps sequences of symbols from an alphabet/vocabulary of indefinite size
    to and from sequential integer ids.

    >>> vec = SequenceVectorizer()
    >>> vec.add_all([['the', 'flat', 'cat', '</s>', '</s>'], ['the', 'cat', 'in', 'the', 'hat']])
    >>> vec.vectorize_all([['in', 'the', 'cat', 'flat', '</s>'],
    ...                    ['the', 'cat', 'sat', '</s>', '</s>']])
    array([[5, 1, 3, 2, 4],
           [1, 3, 0, 4, 4]], dtype=int32)
    >>> vec.unvectorize_all([[1, 3, 0, 5, 1], [1, 2, 3, 6, 4]])
    [['the', 'cat', '<unk>', 'in', 'the'], ['the', 'flat', 'cat', 'hat', '</s>']]
    '''
    def __init__(self):
        self.tokens = []
        self.token_indices = {}
        self.indices_token = {}
        self.max_len = 0
        self.add(['<unk>'])

    def add_all(self, sequences):
        for seq in sequences:
            self.add(seq)

    def add(self, sequence):
        self.max_len = max(self.max_len, len(sequence))
        for token in sequence:
            if token not in self.token_indices:
                self.token_indices[token] = len(self.tokens)
                self.indices_token[len(self.tokens)] = token
                self.tokens.append(token)

    def vectorize(self, sequence):
        return np.array([(self.token_indices[token] if token in self.token_indices
                          else self.token_indices['<unk>'])
                         for token in sequence], dtype=np.int32)

    def vectorize_all(self, sequences):
        return np.array([self.vectorize(seq) for seq in sequences], dtype=np.int32)

    def unvectorize(self, array):
        if hasattr(array, 'tolist'):
            array = array.tolist()
        return [(self.unvectorize(elem) if isinstance(elem, Sequence)
                 else self.indices_token[elem])
                for elem in array]

    def unvectorize_all(self, sequences):
        # unvectorize already accepts sequences of sequences.
        return self.unvectorize(sequences)


class ColorVectorizer(object):
    '''
    Maps colors to a uniform grid of buckets.
    '''
    RANGES_RGB = (256.0, 256.0, 256.0)
    RANGES_HSV = (360.0, 100.0, 100.0)

    def __init__(self, resolution, hsv=False):
        '''
        :param resolution: A length-3 sequence giving numbers of buckets along each
                           dimension of the RGB grid.
        :param bool hsv: If `True`, buckets will be laid out in a grid in HSV space;
                         otherwise, the grid will be in RGB space. Input and output
                         color spaces can be configured on a per-call basis by
                         using the `hsv` parameter of `vectorize` and `unvectorize`.
        '''
        self.resolution = resolution
        self.num_types = reduce(operator.mul, resolution)
        self.hsv = hsv
        ranges = self.RANGES_HSV if hsv else self.RANGES_RGB
        self.bucket_sizes = tuple(d / r for d, r in zip(ranges, resolution))

    def vectorize(self, color, hsv=None):
        '''
        :param color: An length-3 vector or 1D array-like object containing
                      color coordinates.
        :param bool hsv: If `True`, input is assumed to be in HSV space in the range
                         [0, 360], [0, 100], [0, 100]; if `False`, input should be in RGB
                         space in the range [0, 256). `None` (default) means take the
                         color space from the value given to the constructor.
        :return int: The bucket id for `color`

        >>> ColorVectorizer((2, 2, 2)).vectorize((0, 0, 0))
        0
        >>> ColorVectorizer((2, 2, 2)).vectorize((255, 0, 0))
        4
        >>> ColorVectorizer((2, 2, 2)).vectorize((240, 100, 100), hsv=True)
        ... # HSV (240, 100, 100) = RGB (0, 0, 255)
        1
        >>> ColorVectorizer((2, 2, 2), hsv=True).vectorize((0, 0, 0))
        0
        >>> ColorVectorizer((2, 2, 2), hsv=True).vectorize((240, 0, 0))
        ... # yes, this is also black. Using HSV buckets is a questionable decision.
        4
        >>> ColorVectorizer((2, 2, 2), hsv=True).vectorize((0, 255, 0), hsv=False)
        ... # RGB (0, 255, 0) = HSV (120, 100, 100)
        3
        '''
        if hsv is None:
            hsv = self.hsv

        if hsv and not self.hsv:
            c_hsv = color
            c_rgb_0_1 = colorsys.hsv_to_rgb(*(d * 1.0 / r for d, r in zip(c_hsv, self.RANGES_HSV)))
            color_internal = tuple(int(d * 255.99) for d in c_rgb_0_1)
        elif not hsv and self.hsv:
            c_rgb = color
            c_hsv_0_1 = colorsys.rgb_to_hsv(*(d / 256.0 for d in c_rgb))
            color_internal = tuple(int(d * (r - 0.01)) for d, r in zip(c_hsv_0_1, self.RANGES_HSV))
        else:
            ranges = self.RANGES_HSV if self.hsv else self.RANGES_RGB
            color_internal = tuple(min(d, r - 0.01) for d, r in zip(color, ranges))

        bucket_dims = [int(e // r) for e, r in zip(color_internal, self.bucket_sizes)]
        result = (bucket_dims[0] * self.resolution[1] * self.resolution[2] +
                  bucket_dims[1] * self.resolution[2] +
                  bucket_dims[2])
        assert (0 <= result < self.num_types), (color, result)
        return result

    def vectorize_all(self, colors, hsv=None):
        '''
        :param colors: A sequence of length-3 vectors or 1D array-like objects containing
                      RGB coordinates in the range [0, 256).
        :param random: If true, sample a random color from the bucket
        :param bool hsv: If `True`, input is assumed to be in HSV space in the range
                         [0, 360], [0, 100], [0, 100]; if `False`, input should be in RGB
                         space in the range [0, 256). `None` (default) means take the
                         color space from the value given to the constructor.
        :return array(int32): The bucket ids for each color in `colors`

        >>> ColorVectorizer((2, 2, 2)).vectorize_all([(0, 0, 0), (255, 0, 0)])
        array([0, 4], dtype=int32)
        '''
        return np.array([self.vectorize(c, hsv=hsv) for c in colors], dtype=np.int32)

    def unvectorize(self, bucket, random=False, hsv=None):
        '''
        :param int bucket: The id of a color bucket
        :param random: If `True`, sample a random color from the bucket. Otherwise,
                       return the center of the bucket.
        :param hsv: If `True`, return colors in HSV format [0 <= hue <= 360,
                    0 <= sat <= 100, 0 <= val <= 100]; if `False`, RGB
                    [0 <= r/g/b <= 256]. `None` (default) means take the
                    color space from the value given to the constructor.
        :return tuple(int): A color from the bucket with id `bucket`.

        >>> ColorVectorizer((2, 2, 2)).unvectorize(0)
        (64, 64, 64)
        >>> ColorVectorizer((2, 2, 2)).unvectorize(4)
        (192, 64, 64)
        >>> ColorVectorizer((2, 2, 2)).unvectorize(4, hsv=True)
        (0, 66, 75)
        >>> ColorVectorizer((2, 2, 2), hsv=True).unvectorize(0)
        (90, 25, 25)
        >>> ColorVectorizer((2, 2, 2), hsv=True).unvectorize(4)
        (270, 25, 25)
        >>> ColorVectorizer((2, 2, 2), hsv=True).unvectorize(4, hsv=False)
        (56, 48, 64)
        '''
        if hsv is None:
            hsv = self.hsv
        bucket_start = (
            (bucket / (self.resolution[1] * self.resolution[2]) % self.resolution[0]),
            (bucket / self.resolution[2]) % self.resolution[1],
            bucket % self.resolution[2],
        )
        color = tuple((rng.randint(d * size, (d + 1) * size) if random
                       else (d * size + size // 2))
                      for d, size in zip(bucket_start, self.bucket_sizes))
        if self.hsv:
            c_hsv = tuple(int(d) for d in color)
            c_rgb_0_1 = colorsys.hsv_to_rgb(*(d * 1.0 / r for d, r in zip(color, self.RANGES_HSV)))
            c_rgb = tuple(int(d * 256.0) for d in c_rgb_0_1)
        else:
            c_rgb = tuple(int(d) for d in color)
            c_hsv_0_1 = colorsys.rgb_to_hsv(*(d / 256.0 for d in color))
            c_hsv = tuple(int(d * r) for d, r in zip(c_hsv_0_1, self.RANGES_HSV))

        if hsv:
            return c_hsv
        else:
            return c_rgb

    def unvectorize_all(self, buckets, random=False, hsv=None):
        '''
        :param Sequence(int) buckets: A sequence of ids of color buckets
        :param random: If true, sample a random color from each bucket. Otherwise,
                       return the center of the bucket.
        :param hsv: If `True`, return colors in HSV format; otherwise, RGB.
                    `None` (default) means take the color space from the value
                    given to the constructor.
        :return list(tuple(int)): One color from each bucket in `buckets`

        >>> ColorVectorizer((2, 2, 2)).unvectorize_all([0, 4])
        [(64, 64, 64), (192, 64, 64)]
        >>> ColorVectorizer((2, 2, 2)).unvectorize_all([0, 4], hsv=True)
        [(0, 0, 25), (0, 66, 75)]
        '''
        return [self.unvectorize(b, random=random, hsv=hsv) for b in buckets]

    def visualize_distribution(self, dist):
        '''
        :param dist: A distribution over the buckets defined by this vectorizer
        :type dist: array-like with shape `(self.num_types,)``
        :return images: `list(`3-D `np.array` with `shape[2] == 3)`, three images
            with the last dimension being the channels (RGB) of cross-sections
            along each axis, showing the strength of the distribution as the
            intensity of the channel perpendicular to the cross-section.

        >>> ColorVectorizer((2, 2, 2)).visualize_distribution([0, 0.25, 0, 0.5,
        ...                                                    0, 0, 0, 0.25])
        ... # doctest: +NORMALIZE_WHITESPACE
        [array([[[  0,  64,  64], [ 85,  64, 192]],
                [[  0, 192,  64], [255, 192, 192]]]),
         array([[[ 64,   0,  64], [ 64, 255, 192]],
                [[192,   0,  64], [192,  85, 192]]]),
         array([[[ 64,  64, 127], [ 64, 192, 255]],
                [[192,  64,   0], [192, 192, 127]]])]
        '''
        dist_3d = np.asarray(dist).reshape(self.resolution)
        # Compute background: RGB/HSV for each bucket along each face with one channel set to 0
        x, y, z = self.bucket_sizes
        ranges = self.RANGES_HSV if self.hsv else self.RANGES_RGB
        rx, ry, rz = ranges
        images = [
            np.array(
                np.meshgrid(0, np.arange(y // 2, ry, y), np.arange(z // 2, rz, z))
            ).squeeze(2).transpose((1, 2, 0)).astype(np.int),
            np.array(
                np.meshgrid(np.arange(x // 2, rx, x), 0, np.arange(z // 2, rz, z))
            ).squeeze(1).transpose((1, 2, 0)).astype(np.int),
            np.array(
                np.meshgrid(np.arange(x // 2, rx, x), np.arange(y // 2, ry, y), 0)
            ).squeeze(3).transpose((2, 1, 0)).astype(np.int),
        ]
        for axis in range(3):
            xsection = dist_3d.sum(axis=axis)
            xsection /= xsection.max()
            if self.hsv:
                im_float = images[axis].astype(np.float) / np.array(self.RANGES_HSV)
                im_float[:, :, axis] = xsection
                images[axis] = (hsv_to_rgb(im_float) *
                                (np.array(self.RANGES_RGB) - 0.01)).astype(np.int)
            else:
                images[axis][:, :, axis] = (xsection * (ranges[axis] - 0.01)).astype(np.int)
        return images


class SimpleLasagneModel(object):
    def __init__(self, input_vars, target_vars, l_out, loss,
                 optimizer, learning_rate=0.001, id=None):
        options = config.options()

        if not isinstance(input_vars, Sequence):
            raise ValueError('input_vars should be a sequence, instead got %s' % (input_vars,))
        if not isinstance(target_vars, Sequence):
            raise ValueError('target_vars should be a sequence, instead got %s' % (input_vars,))

        self.input_vars = input_vars
        self.l_out = l_out
        self.loss = loss
        self.optimizer = optimizer
        self.id = id
        id_tag = (self.id + '/') if self.id else ''
        id_tag_log = (self.id + ': ') if self.id else ''

        params = self.params()
        (monitored,
         train_loss_grads,
         synth_vars) = self.get_train_loss(target_vars, params)
        self.monitored_tags = monitored.keys()

        if options.true_grad_clipping:
            scaled_grads = total_norm_constraint(train_loss_grads, options.true_grad_clipping)
        else:
            scaled_grads = train_loss_grads

        updates = optimizer(scaled_grads, params, learning_rate=learning_rate)
        if not options.no_nan_suppression:
            updates = apply_nan_suppression(updates)

        if options.detect_nans:
            mode = MonitorMode(post_func=detect_nan)
        else:
            mode = None

        if options.verbosity >= 2:
            print(id_tag_log + 'Compiling training function')
        params = input_vars + target_vars + synth_vars
        if options.verbosity >= 6:
            print('params = %s' % (params,))
        self.train_fn = theano.function(params, monitored.values(),
                                        updates=updates, mode=mode,
                                        name=id_tag + 'train', on_unused_input='warn')
        self.visualize_graphs({'loss': monitored['loss']})

        test_prediction = get_output(l_out, deterministic=True)
        if options.verbosity >= 2:
            print(id_tag_log + 'Compiling prediction function')
        if options.verbosity >= 6:
            print('params = %s' % (input_vars,))
        self.predict_fn = theano.function(input_vars, test_prediction, mode=mode,
                                          name=id_tag + 'predict', on_unused_input='ignore')
        self.visualize_graphs({'test_prediction': test_prediction})

    def visualize_graphs(self, monitored):
        options = config.options()
        id_tag = (self.id + '.') if self.id else ''

        if options.run_dir and not options.no_graphviz:
            for tag, graph in monitored.iteritems():
                tag = tag.replace('/', '.')
                pydotprint(graph, outfile=config.get_file_path(id_tag + tag + '.svg'),
                           format='svg', var_with_name_simple=True)

    def params(self):
        return get_all_params(self.l_out, trainable=True)

    def get_train_loss(self, target_vars, params):
        options = config.options()

        assert len(target_vars) == 1
        prediction = get_output(self.l_out)
        mean_loss = self.loss(prediction, target_vars[0]).mean()
        monitored = [('loss', mean_loss)]
        grads = T.grad(mean_loss, params)
        if options.monitor_grads:
            for p, grad in zip(params, grads):
                monitored.append(('grad/' + p.name, grad))
        return OrderedDict(monitored), grads, []

    def fit(self, Xs, ys, batch_size, num_epochs, summary_writer=None, step=0):
        options = config.options()

        if not isinstance(Xs, Sequence):
            raise ValueError('Xs should be a sequence, instead got %s' % (Xs,))
        if not isinstance(ys, Sequence):
            raise ValueError('ys should be a sequence, instead got %s' % (ys,))
        history = OrderedDict((tag, []) for tag in self.monitored_tags)
        id_tag = (self.id + '/') if self.id else ''
        params = self.params()

        progress.start_task('Epoch', num_epochs)
        epoch_start = time.time()
        for epoch in range(num_epochs):
            progress.progress(epoch)
            history_epoch = OrderedDict((tag, []) for tag in self.monitored_tags)
            num_minibatches_approx = len(ys[0]) // batch_size + 1

            progress.start_task('Minibatch', num_minibatches_approx)
            for i, batch in enumerate(self.minibatches(Xs, ys, batch_size, shuffle=True)):
                progress.progress(i)
                if options.verbosity >= 8:
                    print('types: %s' % ([type(v) for t in batch for v in t],))
                    print('shapes: %s' % ([v.shape for t in batch for v in t],))
                inputs, targets, synth = batch
                monitored = self.train_fn(*inputs + targets + synth)
                for tag, value in zip(self.monitored_tags, monitored):
                    history_epoch[tag].append(value)
            progress.end_task()

            for tag, values in history_epoch.items():
                history[tag].append(np.array(values))
                mean_values = np.mean(values, axis=0)
                if len(mean_values.shape) == 0:
                    summary_writer.log_scalar(step + epoch, tag, mean_values)
                else:
                    summary_writer.log_histogram(step + epoch, tag, mean_values)

            if options.monitor_params:
                for param in params:
                    val = param.get_value()
                    tag = 'param/' + param.name
                    if len(val.shape) == 0:
                        summary_writer.log_scalar(step + epoch, tag, val)
                    else:
                        summary_writer.log_histogram(step + epoch, tag, val)

            epoch_end = time.time()
            examples_per_sec = len(ys[0]) / (epoch_end - epoch_start)
            summary_writer.log_scalar(step + epoch,
                                      id_tag + 'examples_per_sec', examples_per_sec)
            epoch_start = epoch_end
        progress.end_task()

        return history

    def predict(self, Xs):
        options = config.options()

        if not isinstance(Xs, Sequence):
            raise ValueError('Xs should be a sequence, instead got %s' % (Xs,))
        id_tag_log = (self.id + ': ') if self.id else ''
        if options.verbosity >= 8:
            print(id_tag_log + 'predict shapes: %s' % [x.shape for x in Xs])
        return self.predict_fn(*Xs)

    def minibatches(self, inputs, targets, batch_size, shuffle=False):
        '''Lifted mostly verbatim from iterate_minibatches in
        https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py'''
        num_examples = len(targets[0])
        assert all(len(X) == num_examples for X in inputs), \
            repr([type(X) for X in inputs] + [type(y) for y in targets])
        assert all(len(y) == num_examples for y in targets), \
            repr([type(X) for X in inputs] + [type(y) for y in targets])
        if shuffle:
            indices = np.arange(num_examples)
            rng.shuffle(indices)
        last_batch = max(0, num_examples - batch_size)
        for start_idx in range(0, last_batch + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield [X[excerpt] for X in inputs], [y[excerpt] for y in targets], []


class NeuralLearner(Learner):
    '''
    A base class for Lasagne-based learners.
    '''

    def __init__(self, color_resolution, hsv=False, id=None):
        super(NeuralLearner, self).__init__()
        if len(color_resolution) == 1:
            color_resolution = color_resolution * 3

        self.seq_vec = SequenceVectorizer()
        self.color_vec = ColorVectorizer(color_resolution, hsv=hsv)
        self.id = id

    def train(self, training_instances):
        options = config.options()

        self.dataset = training_instances
        xs, ys = self._data_to_arrays(training_instances, init_vectorizer=True)
        self._build_model()

        id_tag = (self.id + ': ') if self.id else ''
        if options.verbosity >= 2:
            print(id_tag + 'Training priors')
        self.prior_emp.fit(xs, ys)
        self.prior_smooth.fit(xs, ys)

        if options.verbosity >= 2:
            print(id_tag + 'Training conditional model')
        summary_path = config.get_file_path('losses.tfevents')
        if summary_path:
            writer = summary.SummaryWriter(summary_path)
        else:
            writer = None
        progress.start_task('Iteration', options.train_iters)
        for iteration in range(options.train_iters):
            progress.progress(iteration)
            self.model.fit(xs, ys, batch_size=options.batch_size, num_epochs=options.train_epochs,
                           summary_writer=writer, step=iteration * options.train_epochs)
            if writer is not None:
                self.on_iter_end((iteration + 1) * options.train_epochs, writer)
        progress.end_task()

    def on_iter_end(self, step, writer):
        pass

    def params(self):
        return self.model.params()

    @property
    def num_params(self):
        all_params = self.params()
        return sum(np.prod(p.get_value().shape) for p in all_params)

    def log_prior_emp(self, input_vars):
        return self.prior_emp.apply(input_vars)

    def log_prior_smooth(self, input_vars):
        return self.prior_smooth.apply(input_vars)

    def sample(self, inputs):
        return self.predict(inputs, random=True, verbosity=-6)

    def sample_prior_emp(self, num_samples):
        indices = rng.randint(len(self.dataset), size=num_samples)
        return [self.dataset[i].stripped() for i in indices]

    def sample_joint_emp(self, num_samples=1):
        input_insts = self.sample_prior_emp(num_samples)
        outputs = self.sample(input_insts)
        for inst, out in zip(input_insts, outputs):
            inst.output = out
        return input_insts

    def log_joint_smooth(self, input_vars, target_var):
        return (self.log_prior_smooth(input_vars) -
                self.loss_out(input_vars, target_var))

    def log_joint_emp(self, input_vars, target_var):
        return (self.log_prior_emp(input_vars) -
                self.loss_out(input_vars, target_var))

    def loss_out(self, input_vars=None, target_var=None):
        if input_vars is None:
            input_vars = self.model.input_vars
        if target_var is None:
            target_var = self.model.target_var
        pred = get_output(self.l_out, dict(zip(self.input_layers, input_vars)))
        return self.loss(pred, target_var)

    def __getstate__(self):
        if not hasattr(self, 'model'):
            raise RuntimeError("trying to pickle a model that hasn't been built yet")
        params = self.params()
        return self.seq_vec, self.color_vec, [p.get_value() for p in params]

    def __setstate__(self, state):
        self.seq_vec, self.color_vec, params_state = state
        self._build_model()
        params = self.params()
        for p, value in zip(params, params_state):
            p.set_value(value)
