from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
# from keras.datasets.data_utils import get_file
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
        print('Iteration', iteration)
        history = model.fit(X, y, batch_size=128, nb_epoch=NUM_EPOCHS)
        print('Iteration %d' % iteration)
        yield (np.mean(history.history['loss']), mat_fn(model))

        # start_index = random.randint(0, len(text) - MAXLEN - 1)


def get_log_prob(model, input, output, vec, verbose=False):
    sentence = ['<s>'] * (vec.max_len - len(input)) + input
    x = np.zeros((1, vec.max_len, len(vec.tokens)))
    for t, in_char in enumerate(sentence):
        x[0, t, vec.token_indices[in_char]] = 1.
    preds = model.predict(x, verbose=0)[0]
    if verbose:
        print('preds: %s' % preds)
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
        print(mat)
        get_log_prob(model, ['red'], (255, 0, 0), vec, verbose=True)
        return np.array(mat)
    return thunk


def evaluate(model, vec):
    return
    '''
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = ':{)>8-)='
        # sentence = text[start_index: start_index + MAXLEN]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iteration in range(400):
            x = np.zeros((1, vec.max_len, len(vec.tokens)))
            for t, char in enumerate(sentence):
                x[0, t, vec.token_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = vec.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            if next_char == '\n':
                break
        print('')
    '''


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
    model = Sequential()
    model.add(LSTM(cell_size, return_sequences=True, input_shape=(vec.max_len, len(vec.tokens))))
    model.add(Dropout(0.2))
    model.add(LSTM(cell_size, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(cell_size))
    model.add(Dropout(0.2))
    model.add(Dense(3))

    model.compile(loss='mean_squared_error', optimizer='rmsprop')

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

    print('Evaluate model...')
    evaluate(model, vec)

    ani = ani and None

    print('Done!')


if __name__ == '__main__':
    for labels, listener in [(L_DATA,  True)]:
        path = 'toy_data_color.txt'
        examples = get_examples(path)
        outputs, inputs = zip(*examples)
        train_and_evaluate(path, inputs, outputs, listener=listener)
