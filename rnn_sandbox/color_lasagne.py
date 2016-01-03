# Hybrid of Lasagne MNIST example
#   <https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py>
# and Keras LSTM example
#   <https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py>
# modified to apply to color description understanding.
from __future__ import print_function
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer, get_output, get_all_params
from lasagne.layers.recurrent import LSTMLayer
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import theano
import theano.tensor as T
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.stats
# import random
from collections import namedtuple
from visualization import plot_matrix


STEP = 1
RESOLUTION = (4, 4, 4)
STDEV = 10.0


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
    tokens = ['<s>', '</s>'] + list(set(t for color, desc in examples for t in desc))
    print('total tokens:', len(tokens))
    max_len = max(len(desc) for color, desc in examples)
    print('max len:', max_len)
    token_indices = dict((c, i) for i, c in enumerate(tokens))
    indices_token = dict((i, c) for i, c in enumerate(tokens))

    # TODO: bucket colors
    return Vectorization(tokens, token_indices, indices_token, max_len)


def data_to_arrays_listener(path, listener=False):
    examples = get_examples(path)
    vec = get_vectorization(examples)

    # cut the text in semi-redundant sequences of MAXLEN characters
    sentences = []
    colors = []
    for i, (color, desc) in enumerate(examples):
        s = ['<s>'] * (vec.max_len - len(desc)) + desc
        print('%s -> %s' % (repr(s), repr(color)))
        sentences.append(s)
        colors.append(color)
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), vec.max_len, len(vec.tokens)), dtype=np.bool)
    y = np.zeros((len(sentences), 3), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        for t, token in enumerate(sentence):
            X[i, t, vec.token_indices[token]] = 1
        y[i, :] = colors[i]

    return X, y, vec


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


NUM_ITERS = 80
NUM_EPOCHS = 100


def train(model, X, y, mat_fn):
    print('Training started...')
    # train the model, output generated text after each iteration
    for iteration in range(1, NUM_ITERS):
        print()
        print('-' * 50)
        print('Iteration %d' % iteration)
        history = model.fit(X, y, batch_size=128, nb_epoch=NUM_EPOCHS)
        yield (np.mean(history['loss']), mat_fn(model))


def get_log_prob(model, input, output, vec, verbose=False):
    sentence = ['<s>'] * (vec.max_len - len(input)) + input
    x = np.zeros((1, vec.max_len, len(vec.tokens)))
    for t, in_char in enumerate(sentence):
        x[0, t, vec.token_indices[in_char]] = 1.
    preds = model.predict(x, verbose=0)[0]
    if verbose:
        print('preds for %s: %s' % (input, preds))
    return scipy.stats.multivariate_normal(preds, STDEV).logpdf(output)


'''
S_DATA = [
    [(255, 0, 0), (0, 0, 255)],
    ['red', 'green', 'blue', 'bright red', 'bright green', 'bright blue'],
]
'''
L_DATA = [
    [['red'], ['bright', 'red'], ['blue'], ['bright', 'blue']],
    [(128, 0, 0), (255, 0, 0), (0, 0, 255), (64, 64, 255)],
]
S_DATA = L_DATA


def matrix(inputs, outputs, vec):
    def thunk(model):
        mat = []
        for i in inputs:
            row = []
            total_prob = 0.0
            for o in outputs:
                prob = get_log_prob(model, i, o, vec)
                row.append(prob)
                total_prob += prob
            # row.append(1.0 - total_prob)
            mat.append(np.array(row) / total_prob)
        # print(mat)
        get_log_prob(model, ['red'], (255, 0, 0), vec, verbose=True)
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
        self.loss_ax.set_ylim(0.0, 20000.0)
        self.loss_ax.set_xlim(0, self.maxt)

    def update(self, data):
        y, mat = data
        self.ydata.append(y)
        self.tdata = range(len(self.ydata))
        self.line.set_data(self.tdata, self.ydata)
        plot_matrix(mat, axes=self.mat_ax, show=False,
                    xlabels=self.xlabels, ylabels=self.ylabels)
        return self.line, self.mat_ax


def train_and_evaluate(path, inputs, outputs, listener=False):
    data_to_arrays = data_to_arrays_listener  # if listener else data_to_arrays_speaker
    X, y, vec = data_to_arrays(path, listener=listener)

    cell_size = 20  # 512
    # build the model: 2 stacked LSTM
    print('Build model...')
    input_var = T.tensor3('inputs')
    l_in = InputLayer(shape=(None, vec.max_len, len(vec.tokens)), input_var=input_var)
    l_lstm1 = LSTMLayer(l_in, num_units=cell_size)
    l_lstm1_drop = DropoutLayer(l_lstm1, p=0.2)
    l_lstm2 = LSTMLayer(l_lstm1_drop, num_units=cell_size)
    l_lstm2_drop = DropoutLayer(l_lstm2, p=0.2)
    l_hidden = DenseLayer(l_lstm2_drop, num_units=cell_size, nonlinearity=None)
    l_hidden_drop = DropoutLayer(l_hidden, p=0.2)
    l_out = DenseLayer(l_hidden_drop, num_units=3, nonlinearity=None)

    model = LasagneModel(input_var, l_out, loss=squared_error, optimizer=rmsprop)

    fig, (top, bottom) = plt.subplots(nrows=2)
    graph = DynamicGraph(top, bottom, maxt=NUM_ITERS, xlabels=outputs, ylabels=inputs)

    # pass a generator in "emitter" to produce data for the update func
    print('Build animation...')
    print('graph.update = %s' % repr(graph.update))
    emitter = lambda: train(model, X, y, matrix(inputs, outputs, vec))
    print('emitter = %s' % repr(emitter))
    ani = animation.FuncAnimation(
        fig, graph.update, emitter,
        interval=10, blit=True, repeat=False
    )

    print('Show animation...')
    plt.show()

    ani = ani and None

    print('Done!')


class LasagneModel(object):
    def __init__(self, input_var, l_out, loss, optimizer):
        target_var = T.matrix('targets')

        prediction = get_output(l_out)
        test_prediction = get_output(l_out, deterministic=True)
        mean_loss = squared_error(prediction, target_var).mean()
        params = get_all_params(l_out, trainable=True)
        updates = optimizer(mean_loss, params, learning_rate=0.001)
        self.train_fn = theano.function([input_var, target_var], mean_loss, updates=updates)
        self.predict_fn = theano.function([input_var], test_prediction)

    def fit(self, X, y, batch_size, nb_epoch):
        loss_history = []
        for epoch in range(nb_epoch):
            loss_epoch = []
            for i, batch in enumerate(self.minibatches(X, y, batch_size, shuffle=True)):
                print('Epoch %d of %d minibatch %d\r' % (epoch + 1, nb_epoch, i), end='')
                inputs, targets = batch
                loss_epoch.append(self.train_fn(inputs, targets))
            loss_history.append(loss_epoch)
        print('')
        return {
            'loss': np.array(loss_history)
        }

    def predict(self, X, verbose='ignored'):
        return self.predict_fn(X)

    def minibatches(self, inputs, targets, batch_size, shuffle=False):
        '''Lifted mostly verbatim from iterate_minibatches in
        https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py'''
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        last_batch = max(0, len(inputs) - batch_size)
        for start_idx in range(0, last_batch + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':
    for labels, listener in [(L_DATA,  True)]:
        path = 'toy_data_color.txt'
        examples = get_examples(path)
        outputs, inputs = zip(*examples)
        train_and_evaluate(path, inputs, outputs, listener=listener)
