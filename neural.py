import colorsys
import lasagne
import numpy as np
import operator
import theano
import theano.tensor as T
from collections import Sequence
from lasagne.layers import get_output, get_all_params
from theano.compile import MonitorMode

from bt import config, progress, summary
from bt.learner import Learner
from bt.rng import get_rng

parser = config.get_options_parser()
parser.add_argument('--train_iters', type=int, default=10,
                    help='Number of iterations')
parser.add_argument('--train_epochs', type=int, default=100,
                    help='Number of epochs per iteration')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of examples per minibatch for training and evaluation')
parser.add_argument('--detect_nans', action='store_true',
                    help='If True, throw an error if a non-finite value is detected.')


rng = get_rng()
lasagne.random.set_rng(rng)


def detect_nan(i, node, fn):
    if not isinstance(node.op, T.AllocEmpty):
        for output in fn.outputs:
            if (not isinstance(output[0], np.random.RandomState) and
                    np.isnan(output[0]).any()):
                print('*** NaN detected ***')
                theano.printing.debugprint(node)
                print('Inputs : %s' % [input[0] for input in fn.inputs])
                print('Outputs: %s' % [output[0] for output in fn.outputs])
                raise AssertionError


class SequenceVectorizer(object):
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
    def __init__(self, resolution):
        '''
        :param resolution: A length-3 sequence giving numbers of buckets along each
                           dimension of the RGB grid.
        '''
        self.resolution = resolution
        self.num_types = reduce(operator.mul, resolution)
        self.bucket_sizes = tuple(256 // r for r in resolution)

    def vectorize(self, color):
        '''
        :param color: An length-3 vector or 1D array-like object containing
                      RGB coordinates in the range [0, 256).
        :return int: The bucket id for `color`

        >>> ColorVectorizer((2, 2, 2)).vectorize((0, 0, 0))
        0
        >>> ColorVectorizer((2, 2, 2)).vectorize((255, 0, 0))
        4
        '''
        bucket_dims = [e // r for e, r in zip(color, self.bucket_sizes)]
        return (bucket_dims[0] * self.resolution[1] * self.resolution[2] +
                bucket_dims[1] * self.resolution[2] +
                bucket_dims[2])

    def vectorize_all(self, colors):
        '''
        :param colors: A sequence of length-3 vectors or 1D array-like objects containing
                      RGB coordinates in the range [0, 256).
        :param random: If true, sample a random color from the bucket
        :return array(int32): The bucket ids for each color in `colors`

        >>> ColorVectorizer((2, 2, 2)).vectorize_all([(0, 0, 0), (255, 0, 0)])
        array([0, 4], dtype=int32)
        '''
        return np.array([self.vectorize(c) for c in colors], dtype=np.int32)

    def unvectorize(self, bucket, random=False, hsv=False):
        '''
        :param int bucket: The id of a color bucket
        :param random: If `True`, sample a random color from the bucket. Otherwise,
                       return the center of the bucket.
        :param hsv: If `True`, return colors in HSV format [0 <= hue <= 360,
                    0 <= sat <= 100, 0 <= val <= 100]; otherwise, RGB
                    [0 <= r/g/b <= 256].
        :return tuple(int): A color from the bucket with id `bucket`.

        >>> ColorVectorizer((2, 2, 2)).unvectorize(0)
        (64, 64, 64)
        >>> ColorVectorizer((2, 2, 2)).unvectorize(4)
        (192, 64, 64)
        >>> ColorVectorizer((2, 2, 2)).unvectorize(4, hsv=True)
        (0, 66, 75)
        '''
        bucket_start = (
            (bucket / (self.resolution[1] * self.resolution[2]) % self.resolution[0]),
            (bucket / self.resolution[2]) % self.resolution[1],
            bucket % self.resolution[2],
        )
        rgb = tuple((rng.randint(d * size, (d + 1) * size) if random
                     else (d * size + size // 2))
                    for d, size in zip(bucket_start, self.bucket_sizes))
        if hsv:
            hue, sat, val = colorsys.rgb_to_hsv(*(d / 256.0 for d in rgb))
            return (int(hue * 360.0), int(sat * 100.0), int(val * 100.0))
        else:
            return rgb

    def unvectorize_all(self, buckets, random=False, hsv=False):
        '''
        :param Sequence(int) buckets: A sequence of ids of color buckets
        :param random: If true, sample a random color from each bucket. Otherwise,
                       return the center of the bucket.
        :param hsv: If `True`, return colors in HSV format; otherwise, RGB.
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
        # Compute background: RGB for each bucket along each face with one channel set to 0
        r, g, b = self.bucket_sizes
        images = [
            np.array(
                np.meshgrid(0, np.arange(g // 2, 256, g), np.arange(b // 2, 256, b))
            ).squeeze(2).transpose((1, 2, 0)),
            np.array(
                np.meshgrid(np.arange(r // 2, 256, r), 0, np.arange(b // 2, 256, b))
            ).squeeze(1).transpose((1, 2, 0)),
            np.array(
                np.meshgrid(np.arange(r // 2, 256, r), np.arange(g // 2, 256, g), 0)
            ).squeeze(3).transpose((2, 1, 0)),
        ]
        for axis in range(3):
            xsection = dist_3d.sum(axis=axis)
            xsection /= xsection.max()
            images[axis][:, :, axis] = (xsection * 255.99).astype(np.int)
        return images


class SimpleLasagneModel(object):
    def __init__(self, input_vars, target_vars, l_out, loss, optimizer, id=None):
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
        (train_loss,
         train_loss_grads,
         synth_vars) = self.get_train_loss(target_vars, params)
        updates = optimizer(train_loss_grads, params, learning_rate=0.001)
        if options.detect_nans:
            mode = MonitorMode(post_func=detect_nan)
        else:
            mode = None
        print(id_tag_log + 'Compiling training function')
        params = input_vars + target_vars + synth_vars
        print('params = %s' % (params,))
        self.train_fn = theano.function(params,
                                        train_loss, updates=updates, mode=mode,
                                        name=id_tag + 'train', on_unused_input='warn')

        test_prediction = get_output(l_out, deterministic=True)
        print(id_tag_log + 'Compiling prediction function')
        print('params = %s' % (input_vars,))
        self.predict_fn = theano.function(input_vars, test_prediction, mode=mode,
                                          name=id_tag + 'predict', on_unused_input='ignore')

    def params(self):
        return get_all_params(self.l_out, trainable=True)

    def get_train_loss(self, target_vars, params):
        assert len(target_vars) == 1
        prediction = get_output(self.l_out)
        mean_loss = self.loss(prediction, target_vars[0]).mean()
        return mean_loss, T.grad(mean_loss, params), []

    def fit(self, Xs, ys, batch_size, num_epochs):
        if not isinstance(Xs, Sequence):
            raise ValueError('Xs should be a sequence, instead got %s' % (Xs,))
        if not isinstance(ys, Sequence):
            raise ValueError('ys should be a sequence, instead got %s' % (ys,))
        loss_history = []

        progress.start_task('Epoch', num_epochs)
        for epoch in range(num_epochs):
            progress.progress(epoch)
            loss_epoch = []
            num_minibatches_approx = len(ys[0]) // batch_size + 1

            progress.start_task('Minibatch', num_minibatches_approx)
            for i, batch in enumerate(self.minibatches(Xs, ys, batch_size, shuffle=True)):
                progress.progress(i)
                print('types: %s' % ([type(v) for t in batch for v in t],))
                print('shapes: %s' % ([v.shape for t in batch for v in t],))
                inputs, targets, synth = batch
                loss_epoch.append(self.train_fn(*inputs + targets + synth))
            progress.end_task()

            loss_history.append(loss_epoch)
        progress.end_task()

        return np.array(loss_history)

    def predict(self, Xs):
        if not isinstance(Xs, Sequence):
            raise ValueError('Xs should be a sequence, instead got %s' % (Xs,))
        id_tag_log = (self.id + ': ') if self.id else ''
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

    def __init__(self, color_resolution, id=None):
        super(NeuralLearner, self).__init__()
        res = color_resolution

        self.seq_vec = SequenceVectorizer()
        self.color_vec = ColorVectorizer((res, res, res))
        self.id = id

    def train(self, training_instances):
        options = config.options()

        self.dataset = training_instances
        xs, ys = self._data_to_arrays(training_instances, init_vectorizer=True)
        self._build_model()

        id_tag = (self.id + ': ') if self.id else ''
        print(id_tag + 'Training priors')
        self.prior_emp.fit(xs, ys)
        self.prior_smooth.fit(xs, ys)

        print(id_tag + 'Training conditional model')
        summary_path = config.get_file_path('losses.tfevents')
        if summary_path:
            writer = summary.SummaryWriter(summary_path)
        progress.start_task('Iteration', options.train_iters)
        for iteration in range(options.train_iters):
            progress.progress(iteration)
            losses_iter = self.model.fit(xs, ys, batch_size=options.batch_size,
                                         num_epochs=options.train_epochs)
            if summary_path:
                for e, loss in enumerate(np.mean(losses_iter, axis=1).tolist()):
                    writer.log_scalar(iteration * options.train_epochs + e,
                                      'loss_epoch', loss)
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
        raise NotImplementedError

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
        l_out, loss = self._get_l_out(input_vars)
        return (self.log_prior_smooth(input_vars) -
                loss(get_output(l_out), target_var))

    def log_joint_emp(self, input_vars, target_var):
        l_out, loss = self._get_l_out(input_vars)
        return (self.log_prior_emp(input_vars) -
                loss(get_output(l_out), target_var))

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
