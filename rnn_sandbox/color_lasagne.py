# Hybrid of Lasagne MNIST example
#   <https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py>
# and Keras LSTM example
#   <https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py>
# modified to apply to color description understanding.
from __future__ import print_function
import operator
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, EmbeddingLayer
from lasagne.layers import ReshapeLayer, NonlinearityLayer, ConcatLayer
from lasagne.layers import get_output, get_all_params
from lasagne.layers.recurrent import LSTMLayer
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from lasagne.updates import rmsprop
import theano
import theano.tensor as T
from theano.tensor.nnet import crossentropy_categorical_1hot
from theano.compile import MonitorMode
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# import random
from collections import namedtuple, Sequence
from visualization import plot_matrix


STEP = 1
RESOLUTION = (4, 4, 4)
NUM_BUCKETS = reduce(operator.mul, RESOLUTION)
BUCKET_SIZES = tuple(256 // r for r in RESOLUTION)
STDEV = 10.0

DETECT_NANS = False


Vectorization = namedtuple('Vectorization', (
    'tokens',
    'token_indices',
    'indices_token',
    'max_len',
))


def get_examples(path):
    examples = []
    with open(path, 'r') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 3:
                raise ValueError('Not enough columns in data file line: %s' % repr(line))
            try:
                color = tuple(float(t) for t in tokens[:3])
            except ValueError:
                raise ValueError('Invalid float format in data file line: %s' % repr(line))
            desc = tokens[3:]

            examples.append((color, desc))
    print('number of examples:', len(examples))
    return examples


def get_vectorization(examples):
    tokens = ['<MASK>', '<s>', '</s>'] + list(set(t for color, desc in examples for t in desc))
    print('total tokens:', len(tokens))
    max_len = max(len(desc) for color, desc in examples) + 2
    print('max len (including <s> </s>):', max_len)
    token_indices = dict((c, i) for i, c in enumerate(tokens))
    indices_token = dict((i, c) for i, c in enumerate(tokens))

    # TODO: bucket colors
    return Vectorization(tokens, token_indices, indices_token, max_len)


def get_bucket(color):
    bucket_dims = [e // r for e, r in zip(color, BUCKET_SIZES)]
    return (bucket_dims[0] * RESOLUTION[1] * RESOLUTION[2] +
            bucket_dims[1] * RESOLUTION[2] +
            bucket_dims[2])


def get_bucket_dims(bucket):
    return (
        (bucket / (RESOLUTION[1] * RESOLUTION[2]) % RESOLUTION[0]),
        (bucket / RESOLUTION[2]) % RESOLUTION[1],
        bucket % RESOLUTION[2],
    )


def sample_from_bucket(bucket):
    bucket_dims = get_bucket_dims(bucket)
    return tuple(np.random.randint(d * size, (d + 1) * size)
                 for d, size in zip(bucket_dims, BUCKET_SIZES))


def strip_invalid_tokens(sentence):
    good_tokens = [t for t in sentence if t not in ('<s>', '<MASK>')]
    if '</s>' in good_tokens:
        end_pos = good_tokens.index('</s>')
        good_tokens = good_tokens[:end_pos]
    return good_tokens


def sample(a, temperature=1.0):
    a = np.array(a)
    if len(a.shape) < 1:
        raise ValueError('scalar is not a valid probability distribution')
    elif len(a.shape) == 1:
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))
    else:
        return np.array([sample(s, temperature) for s in a])


NUM_ITERS = 80
NUM_EPOCHS = 100


def train(agent, X, y, mat_fn):
    print('Training started...')
    # train the model, output generated text after each iteration
    for iteration in range(1, NUM_ITERS):
        print()
        print('-' * 50)
        print('Iteration %d' % iteration)
        history = agent.model.fit(X, y, batch_size=128, nb_epoch=NUM_EPOCHS)
        mean_loss = np.mean(history['loss'])
        print('Mean loss: %s' % mean_loss)
        print('Samples:')
        inputs = agent.sample_prior(10)
        for inp, out in zip(inputs, agent.sample(inputs)):
            print('%s -> %s' % (inp, out))
        yield (mean_loss, mat_fn(agent.model))


'''
S_DATA = [
    [(255, 0, 0), (0, 0, 255)],
    [['red'], ['green'], ['blue'],
     ['bright', 'red'], ['bright', 'green'], ['bright', 'blue']],
]
S_DATA = [
    [(128, 0, 0), (255, 0, 0), (0, 0, 255), (64, 64, 255)],
    [['red'], ['bright', 'red'], ['blue'], ['bright', 'blue']],
]
L_DATA = [
    [['red'], ['bright', 'red'], ['blue'], ['bright', 'blue']],
    [(128, 0, 0), (255, 0, 0), (0, 0, 255), (64, 64, 255)],
]
'''
S_DATA = [
    [(250, 2, 4), (0, 255, 39), (2, 128, 0), (44, 33, 255)],
    [['red'], ['bright', 'green'], ['green'], ['bright', 'blue']],
]
L_DATA = [
    [['red'], ['bright', 'green'], ['green'], ['bright', 'blue']],
    [(250, 2, 4), (0, 255, 39), (2, 128, 0), (44, 33, 255)],
]


def matrix(inputs, outputs, vec, get_log_prob):
    def thunk(model):
        mat = []
        est_diag_log_prob = 0.0
        for i, inp in enumerate(inputs):
            row = []
            total_prob = 0.0
            for j, out in enumerate(outputs):
                prob = get_log_prob(model, inp, out, vec)
                row.append(prob)
                total_prob += prob
                if i == j:
                    est_diag_log_prob += np.log(prob)
            # row.append(1.0 - total_prob)
            # mat.append(np.array(row) / total_prob)
            mat.append(np.array(row))
        # print(mat)
        get_log_prob(model, inputs[0], outputs[0], vec, verbose=True)
        print('Estimated diagonal log prob: %s' % est_diag_log_prob)
        return np.array(mat)
    return thunk


class DynamicGraph(object):
    def __init__(self, loss_ax, mat_ax, maxt=2, dt=0.02, xlabels=[], ylabels=[]):
        self.loss_ax = loss_ax
        self.mat_ax = mat_ax
        self.dt = dt
        self.maxt = maxt
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.tdata = []
        self.ydata = []
        self.line = Line2D(self.tdata, self.ydata)
        self.loss_ax.add_line(self.line)
        self.loss_ax.set_xlim(0, self.maxt)

    def update(self, data):
        y, mat = data
        self.ydata.append(y)
        lim_low = min(0.0, min(self.ydata))
        lim_hi = max(0.0, max(self.ydata))
        self.loss_ax.set_ylim(lim_low, lim_hi)
        self.tdata = range(len(self.ydata))
        self.line.set_data(self.tdata, self.ydata)
        plot_matrix(mat, axes=self.mat_ax, show=False,
                    xlabels=self.xlabels, ylabels=self.ylabels)
        return self.line, self.mat_ax


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
    return crossentropy_categorical_1hot(T.reshape(coding_dist, (-1, T.shape(coding_dist)[-1])),
                                         true_idx.flatten())


def train_and_evaluate(path, inputs, outputs, listener=False):
    agent = ListenerModel() if listener else SpeakerModel()
    Xs, y, agent.vec = agent.data_to_arrays(path)

    # build the model: 2 stacked LSTM
    print('Build model...')
    agent.model = agent.build_model(agent.vec)

    fig, (top, bottom) = plt.subplots(nrows=2)
    graph = DynamicGraph(top, bottom, maxt=NUM_ITERS, xlabels=outputs, ylabels=inputs)

    # pass a generator in "emitter" to produce data for the update func
    print('Build animation...')
    print('graph.update = %s' % repr(graph.update))
    emitter = lambda: train(agent, Xs, y, matrix(inputs, outputs, agent.vec, agent.get_log_prob))
    print('emitter = %s' % repr(emitter))
    ani = animation.FuncAnimation(
        fig, graph.update, emitter,
        interval=10, blit=True, repeat=False
    )

    print('Show animation...')
    plt.show()

    ani = ani and None

    print('Done!')


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


class LasagneModel(object):
    def __init__(self, input_vars, target_var, l_out, loss, optimizer):
        if not isinstance(input_vars, Sequence):
            input_vars = [input_vars]

        prediction = get_output(l_out)
        test_prediction = get_output(l_out, deterministic=True)
        mean_loss = loss(prediction, target_var).mean()
        params = get_all_params(l_out, trainable=True)
        updates = optimizer(mean_loss, params, learning_rate=0.001)
        if DETECT_NANS:
            mode = MonitorMode(post_func=detect_nan)
        else:
            mode = None
        self.train_fn = theano.function(input_vars + [target_var], mean_loss, updates=updates,
                                        mode=mode)
        self.predict_fn = theano.function(input_vars, test_prediction, mode=mode)

    def fit(self, Xs, y, batch_size, nb_epoch):
        if not isinstance(Xs, Sequence):
            Xs = [Xs]
        loss_history = []
        for epoch in range(nb_epoch):
            loss_epoch = []
            for i, batch in enumerate(self.minibatches(Xs, y, batch_size, shuffle=True)):
                print('Epoch %d of %d minibatch %d\r' % (epoch + 1, nb_epoch, i), end='')
                inputs, targets = batch
                loss_epoch.append(self.train_fn(*inputs + [targets]))
            loss_history.append(loss_epoch)
        print('')
        return {
            'loss': np.array(loss_history)
        }

    def predict(self, Xs, verbose='ignored'):
        if not isinstance(Xs, Sequence):
            Xs = [Xs]
        return self.predict_fn(*Xs)

    def minibatches(self, inputs, targets, batch_size, shuffle=False):
        '''Lifted mostly verbatim from iterate_minibatches in
        https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py'''
        assert all(len(X) == len(targets) for X in inputs)
        if shuffle:
            indices = np.arange(len(targets))
            np.random.shuffle(indices)
        last_batch = max(0, len(targets) - batch_size)
        for start_idx in range(0, last_batch + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield [X[excerpt] for X in inputs], targets[excerpt]


class ListenerModel(object):
    def data_to_arrays(self, path):
        examples = get_examples(path)
        vec = get_vectorization(examples)

        sentences = []
        colors = []
        for i, (color, desc) in enumerate(examples):
            s = ['<s>'] * (vec.max_len - 1 - len(desc)) + desc
            s.append('</s>')
            print('%s -> %s' % (repr(s), repr(color)))
            sentences.append(s)
            colors.append(color)
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        X = np.zeros((len(sentences), vec.max_len), dtype=np.int32)
        y = np.zeros((len(sentences),), dtype=np.int32)
        for i, sentence in enumerate(sentences):
            for t, token in enumerate(sentence):
                X[i, t] = vec.token_indices[token]
            y[i] = get_bucket(colors[i])

        print('X:')
        print(X)
        print('y:')
        print(y)
        return X, y, vec

    def build_model(self, vec):
        cell_size = 20  # 512
        input_var = T.imatrix('inputs')
        target_var = T.ivector('targets')

        l_in = InputLayer(shape=(None, vec.max_len), input_var=input_var)
        l_in_embed = EmbeddingLayer(l_in, input_size=len(vec.tokens), output_size=cell_size)
        l_lstm1 = LSTMLayer(l_in_embed, num_units=cell_size)
        l_lstm1_drop = DropoutLayer(l_lstm1, p=0.2)
        l_lstm2 = LSTMLayer(l_lstm1_drop, num_units=cell_size)
        l_lstm2_drop = DropoutLayer(l_lstm2, p=0.2)

        l_hidden = DenseLayer(l_lstm2_drop, num_units=cell_size, nonlinearity=None)
        l_hidden_drop = DropoutLayer(l_hidden, p=0.2)
        l_scores = DenseLayer(l_hidden_drop, num_units=NUM_BUCKETS, nonlinearity=None)
        l_out = NonlinearityLayer(l_scores, nonlinearity=softmax)

        return LasagneModel(input_var, target_var, l_out,
                            loss=categorical_crossentropy, optimizer=rmsprop)

    def get_predictions(self, model, inputs, vec):
        x = np.zeros((len(inputs), vec.max_len), dtype=np.int32)
        for i, inp in enumerate(inputs):
            sentence = ['<s>'] * (vec.max_len - 1 - len(inp)) + inp + ['</s>']
            for t, in_token in enumerate(sentence):
                x[i, t] = vec.token_indices[in_token]
        return model.predict(x, verbose=0)

    def get_log_prob(self, model, input, output, vec, verbose=False):
        preds = self.get_predictions(model, [input], vec)
        # if verbose:
        #     print('preds for %s: %s' % (input, preds))
        return preds[0, get_bucket(output)]

    def sample(self, inputs):
        preds = self.get_predictions(self.model, inputs, self.vec)
        buckets = sample(preds)
        return [sample_from_bucket(b) for b in buckets]

    def sample_prior(self, num_samples):
        # TODO: language model
        utterances = L_DATA[0]
        return [utterances[np.random.randint(len(utterances))]
                for _ in range(num_samples)]


class SpeakerModel(object):
    def data_to_arrays(self, path):
        examples = get_examples(path)
        vec = get_vectorization(examples)

        colors = []
        previous = []
        next_tokens = []
        for i, (color, desc) in enumerate(examples):
            full = ['<s>'] + desc + ['</s>'] + ['<MASK>'] * (vec.max_len - 2 - len(desc))
            prev = full[:-1]
            next = full[1:]
            print('%s, %s -> %s' % (repr(color), repr(prev), repr(next)))
            colors.append(color)
            previous.append(prev)
            next_tokens.append(next)
        print('number of sequences:', len(colors))

        print('Vectorization...')
        c = np.zeros((len(colors),), dtype=np.int32)
        P = np.zeros((len(previous), vec.max_len - 1), dtype=np.int32)
        mask = np.zeros((len(previous), vec.max_len - 1), dtype=np.int32)
        N = np.zeros((len(next_tokens), vec.max_len - 1), dtype=np.int32)
        for i, (color, prev, next) in enumerate(zip(colors, previous, next_tokens)):
            c[i] = get_bucket(colors[i])
            for t, token in enumerate(prev):
                P[i, t] = vec.token_indices[token]
            for t, token in enumerate(next):
                N[i, t] = vec.token_indices[token]
                mask[i, t] = (token != '<MASK>')
        c = np.tile(c[:, np.newaxis], [1, vec.max_len - 1])

        print('c:')
        print(c)
        print('P:')
        print(P)
        print('mask:')
        print(mask)
        print('N:')
        print(N)
        return (c, P, mask), N, vec

    def build_model(self, vec):
        cell_size = 20  # 512
        input_var = T.imatrix('inputs')
        prev_output_var = T.imatrix('previous')
        mask_var = T.imatrix('mask')
        target_var = T.imatrix('targets')

        '''
        l_in = InputLayer(shape=(None, 3), input_var=input_var)
        l_hidden1 = DenseLayer(l_in, num_units=cell_size, nonlinearity=tanh)
        l_hidden1_drop = DropoutLayer(l_hidden1, p=0.2)
        l_hidden2 = DenseLayer(l_hidden1_drop, num_units=cell_size, nonlinearity=tanh)
        l_hidden2_drop = DropoutLayer(l_hidden2, p=0.2)
        '''

        l_color = InputLayer(shape=(None, vec.max_len - 1), input_var=input_var)
        l_color_embed = EmbeddingLayer(l_color, input_size=NUM_BUCKETS, output_size=cell_size)
        l_prev_out = InputLayer(shape=(None, vec.max_len - 1),
                                input_var=prev_output_var)
        l_prev_embed = EmbeddingLayer(l_prev_out, input_size=len(vec.tokens), output_size=cell_size)
        l_in = ConcatLayer([l_color_embed, l_prev_embed], axis=2)
        l_mask_in = InputLayer(shape=(None, vec.max_len - 1),
                               input_var=mask_var)
        # l_lstm1 = LSTMLayer(l_in, cell_init=l_hidden2_drop,
        #                     mask_input=l_mask_in, num_units=cell_size)
        l_lstm1 = LSTMLayer(l_in, mask_input=l_mask_in, num_units=cell_size)
        l_lstm1_drop = DropoutLayer(l_lstm1, p=0.2)
        l_lstm2 = LSTMLayer(l_lstm1_drop, num_units=len(vec.tokens))
        l_shape = ReshapeLayer(l_lstm2, (-1, len(vec.tokens)))
        l_softmax = NonlinearityLayer(l_shape, nonlinearity=softmax)
        l_out = ReshapeLayer(l_softmax, (-1, vec.max_len - 1, len(vec.tokens)))

        return LasagneModel([input_var, prev_output_var, mask_var], target_var, l_out,
                            loss=crossentropy_categorical_1hot_nd, optimizer=rmsprop)

    def get_predictions(self, inputs, prevs, lengths=None, verbose=False):
        if lengths is None:
            lengths = [len(p) + 1 for p in prevs]
        c = np.zeros((len(inputs), self.vec.max_len - 1), dtype=np.int32)
        P = np.zeros((len(inputs), self.vec.max_len - 1), dtype=np.int32)
        mask = np.zeros((len(inputs), self.vec.max_len - 1), dtype=np.int32)
        for i, (inp, prev, length) in enumerate(zip(inputs, prevs, lengths)):
            c[i, :] = get_bucket(inp)
            for t, in_token in enumerate(prev):
                P[i, t] = self.vec.token_indices[in_token]
                mask[i, t] = (t <= length)
        if verbose and False:
            print('c:')
            print(c)
            print('P:')
            print(P)
            print('mask:')
            print(mask)
        return self.model.predict([c, P, mask], verbose=0)

    def get_log_prob(self, model, input, output, vec, verbose=False):
        full = ['<s>'] + output + ['</s>'] + ['<MASK>'] * (vec.max_len - 2 - len(output))
        prev = full[:-1]
        next = full[1:]
        if verbose:
            print('input = %s, output = %s' % (input, output))
        preds = self.get_predictions([input], prevs=[prev], lengths=[len(output)], verbose=verbose)
        # if verbose:
        #     print('preds for %s: %s' % (input, preds))
        log_prob = 1.0
        for t, out_token in enumerate(next):
            if out_token != '<MASK>':
                log_prob *= preds[0, t, vec.token_indices[out_token]]
        return log_prob

    def sample(self, inputs):
        done = np.zeros((len(inputs),), dtype=np.bool)
        outputs = [['<s>'] for _ in inputs]
        length = 0
        while not done.all() and length < self.vec.max_len - 1:
            preds = self.get_predictions(inputs, prevs=outputs)
            indices = sample(preds[:, length, :])
            for out, idx in zip(outputs, indices):
                out.append(self.vec.indices_token[idx])
            done = np.logical_or(done, indices == self.vec.token_indices['</s>'])
            length += 1
        return [strip_invalid_tokens(o) for o in outputs]

    def sample_prior(self, num_samples):
        # return [tuple(np.random.randint(0, 255) for _ in range(3))
        #         for _ in range(num_samples)]
        colors = S_DATA[0]
        return [colors[np.random.randint(len(colors))]
                for _ in range(num_samples)]


def main():
    for labels, listener in [(L_DATA, True), (S_DATA, False)]:
        path = 'toy_data_color.txt'
        inputs, outputs = labels
        train_and_evaluate(path, inputs, outputs, listener=listener)


if __name__ == '__main__':
    main()
