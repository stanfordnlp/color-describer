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
from bt.instance import Instance
from bt.rng import get_rng

parser = config.get_options_parser()
parser.add_argument('--train_iters', type=int, default=10,
                    help='Number of iterations')
parser.add_argument('--train_epochs', type=int, default=100,
                    help='Number of epochs per iteration')
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

    def unvectorize(self, bucket, random=False):
        '''
        :param int bucket: The id of a color bucket
        :param random: If true, sample a random color from the bucket. Otherwise,
                       return the center of the bucket.
        :return tuple(int): A color from the bucket with id `bucket`.

        >>> ColorVectorizer((2, 2, 2)).unvectorize(0)
        (64, 64, 64)
        >>> ColorVectorizer((2, 2, 2)).unvectorize(4)
        (192, 64, 64)
        '''
        bucket_start = (
            (bucket / (self.resolution[1] * self.resolution[2]) % self.resolution[0]),
            (bucket / self.resolution[2]) % self.resolution[1],
            bucket % self.resolution[2],
        )
        return tuple((rng.randint(d * size, (d + 1) * size) if random
                      else (d * size + size // 2))
                     for d, size in zip(bucket_start, self.bucket_sizes))

    def unvectorize_all(self, buckets, random=False):
        '''
        :param Sequence(int) buckets: A sequence of ids of color buckets
        :param random: If true, sample a random color from each bucket. Otherwise,
                       return the center of the bucket.
        :return list(tuple(int)): One color from each bucket in `buckets`

        >>> ColorVectorizer((2, 2, 2)).unvectorize_all([0, 4])
        [(64, 64, 64), (192, 64, 64)]
        '''
        return [self.unvectorize(b, random=random) for b in buckets]


class SimpleLasagneModel(object):
    def __init__(self, input_vars, target_vars, l_out, loss, optimizer):
        options = config.options()

        if not isinstance(input_vars, Sequence):
            input_vars = [input_vars]
        if not isinstance(target_vars, Sequence):
            target_vars = [target_vars]

        self.l_out = l_out
        self.loss = loss

        params = get_all_params(l_out, trainable=True)
        (train_loss,
         train_loss_grads,
         synth_vars) = self.get_train_loss(target_vars, params)
        updates = optimizer(train_loss_grads, params, learning_rate=0.001)
        if options.detect_nans:
            mode = MonitorMode(post_func=detect_nan)
        else:
            mode = None
        print('Compiling training function')
        self.train_fn = theano.function(input_vars + target_vars + synth_vars,
                                        train_loss, updates=updates, mode=mode)

        test_prediction = get_output(l_out, deterministic=True)
        print('Compiling prediction function')
        self.predict_fn = theano.function(input_vars, test_prediction, mode=mode)

    def get_train_loss(self, target_vars, params):
        assert len(target_vars) == 1
        prediction = get_output(self.l_out)
        mean_loss = self.loss(prediction, target_vars[0]).mean()
        return mean_loss, T.grad(mean_loss, params), []

    def fit(self, Xs, ys, batch_size, num_epochs):
        if not isinstance(Xs, Sequence):
            Xs = [Xs]
        if not isinstance(ys, Sequence):
            ys = [ys]
        loss_history = []

        progress.start_task('Epoch', num_epochs)
        for epoch in range(num_epochs):
            progress.progress(epoch)
            loss_epoch = []
            num_minibatches_approx = len(ys[0]) // batch_size + 1

            progress.start_task('Minibatch', num_minibatches_approx)
            for i, batch in enumerate(self.minibatches(Xs, ys, batch_size, shuffle=True)):
                progress.progress(i)
                inputs, targets, synth = batch
                loss_epoch.append(self.train_fn(*inputs + targets + synth))
            progress.end_task()

            loss_history.append(loss_epoch)
        progress.end_task()

        return np.array(loss_history)

    def predict(self, Xs):
        if not isinstance(Xs, Sequence):
            Xs = [Xs]
        return self.predict_fn(*Xs)

    def minibatches(self, inputs, targets, batch_size, shuffle=False):
        '''Lifted mostly verbatim from iterate_minibatches in
        https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py'''
        num_examples = len(targets[0])
        assert all(len(X) == num_examples for X in inputs)
        assert all(len(y) == num_examples for y in targets)
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

    def __init__(self, color_resolution):
        super(NeuralLearner, self).__init__()
        res = color_resolution

        self.seq_vec = SequenceVectorizer()
        self.color_vec = ColorVectorizer((res, res, res))

    def train(self, training_instances):
        options = config.options()

        self.dataset = training_instances
        xs, ys = self._data_to_arrays(training_instances)
        self._build_model()

        print('Training')
        loss_writer = summary.SummaryWriter(config.get_file_path('losses.tfevents'))
        progress.start_task('Iteration', options.train_iters)
        for iteration in range(options.train_iters):
            progress.progress(iteration)
            losses_iter = self.model.fit(xs, ys, batch_size=128, num_epochs=options.train_epochs)
            for e, loss in enumerate(np.mean(losses_iter, axis=1).tolist()):
                loss_writer.log_scalar(iteration * options.train_epochs + e,
                                       'loss_epoch', loss)
        progress.end_task()

    def params(self):
        return get_all_params(self.model.l_out)

    def log_prior_emp(self, input_vars):
        raise NotImplementedError

    def log_prior_smooth(self, input_vars):
        raise NotImplementedError

    def sample(self, inputs):
        raise NotImplementedError

    def sample_prior_emp(self, num_samples):
        indices = rng.randint(len(self.dataset), size=num_samples)
        return [self.dataset[i].stripped() for i in indices]

    def sample_joint_emp(self, num_samples=1):
        inputs = self.sample_prior_emp(num_samples)
        outputs = self.sample(inputs)
        return [Instance(input=inp, output=out) for inp, out in zip(inputs, outputs)]

    def log_joint_smooth(self, input_vars, target_var):
        return (self.log_prior_smooth(input_vars) -
                self.model.loss(get_output(self.model._get_l_out(input_vars)),
                                target_var))

    def log_joint_emp(self, input_vars, target_var):
        return (self.log_prior_emp(input_vars) -
                self.model.loss(get_output(self.model._get_l_out(input_vars)),
                                target_var))

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
