import lasagne
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as G
import time
from collections import Sequence, OrderedDict
from lasagne.layers import get_output, get_all_params
from lasagne.updates import total_norm_constraint
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
parser.add_argument('--true_grad_clipping', type=float, default=5.0,
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


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.array(a)
    if len(a.shape) < 1:
        raise ValueError('scalar is not a valid probability distribution')
    elif len(a.shape) == 1:
        # Cast to higher resolution to try to get high-precision normalization
        a = np.exp(np.log(a) / temperature).astype(np.float64)
        a /= np.sum(a)
        return np.argmax(rng.multinomial(1, a, 1))
    else:
        return np.array([sample(s, temperature) for s in a])


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
            # TODO: print_mode='all' somehow is always printing, even when
            # there are no NaNs. But tests are passing, even on GPU!
            updates = apply_nan_suppression(updates, print_mode='none')

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
                    if options.verbosity >= 10:
                        print('%s: %s' % (tag, value))
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

    def __init__(self, id=None):
        super(NeuralLearner, self).__init__()
        self.id = id

    def train(self, training_instances, validation_instances=None, metrics=None):
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
            validation_results = self.validate(validation_instances, metrics, iteration=iteration)
            if writer is not None:
                step = (iteration + 1) * options.train_epochs
                self.on_iter_end(step, writer)
                for key, value in validation_results.iteritems():
                    tag = 'val/' + key.split('.', 1)[1].replace('.', '/')
                    writer.log_scalar(step, tag, value)
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
        # TODO: remove references to the vectorizers from this superclass
        return self.seq_vec, self.color_vec, [p.get_value() for p in params], self.id

    def __setstate__(self, state):
        self.unpickle(state)

    def unpickle(self, state, model_class=SimpleLasagneModel):
        # TODO: remove references to the vectorizers from this superclass
        if len(state) == 3:
            self.seq_vec, self.color_vec, params_state = state
            self.id = None
        else:
            self.seq_vec, self.color_vec, params_state, self.id = state
        self._build_model(model_class)
        params = self.params()
        for p, value in zip(params, params_state):
            p.set_value(value)
