from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
# from keras.datasets.data_utils import get_file
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# import random
import sys
from collections import namedtuple
from visualization import plot_matrix

'''
    Example script to generate text from Nietzsche's writings.
    At least 20 epochs are required before the generated text
    starts sounding coherent.
    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.
    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''

# path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")


MAXLEN = 8
STEP = 1


Vectorization = namedtuple('Vectorization', (
    'chars',
    'char_indices',
    'indices_char',
))


def data_to_arrays(path):
    text = open(path).read().lower()
    print('corpus length:', len(text))

    chars = set(text)
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of MAXLEN characters
    sentences = []
    next_chars = []
    for i in range(0, len(text) - MAXLEN, STEP):
        s = text[i: i + MAXLEN]
        nc = text[i + MAXLEN]
        print('%s -> %s' % (repr(s), repr(nc)))
        sentences.append(s)
        next_chars.append(nc)
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return X, y, Vectorization(chars, char_indices, indices_char)


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


NUM_ITERS = 150


def train(model, X, y, mat_fn):
    print('Training started...')
    # train the model, output generated text after each iteration
    for iteration in range(1, NUM_ITERS):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        history = model.fit(X, y, batch_size=128, nb_epoch=1)
        print('Iteration %d' % iteration)
        yield (history.history['loss'][0], mat_fn(model))

        # start_index = random.randint(0, len(text) - MAXLEN - 1)


def get_prob(model, input, output, vec):
    prob = 1.0
    for s, out_char in enumerate(output):
        sentence = (input[len(input) - MAXLEN + s:] +
                    output[max(0, s - MAXLEN):s])
        x = np.zeros((1, MAXLEN, len(vec.chars)))
        for t, in_char in enumerate(sentence):
            x[0, t, vec.char_indices[in_char]] = 1.
        preds = model.predict(x, verbose=0)[0]
        char_prob = preds[vec.char_indices[out_char]]
        prob *= char_prob
        # print('%s -> %s [%f]' % (sentence, out_char, char_prob))
    # print('')
    return prob


S_DATA = [
    [':{)>8-)=', '8-)>:{)='],
    ['\n', 'b\n', 'g\n', 'p\n', 'bg\n', 'bp\n', 'gp\n', 'bgp\n'],
]
L_DATA = [
    [':{)8-)>gp=', ':{)8-)>bp='],
    ['1\n', '2\n'],
]


def matrix(inputs, outputs, vec):
    def thunk(model):
        mat = []
        for i in inputs:
            row = []
            total_prob = 0.0
            for o in outputs:
                prob = get_prob(model, i, o, vec)
                row.append(prob)
                total_prob += prob
            row.append(1.0 - total_prob)
            mat.append(row)
        print(mat)
        return np.array(mat)
    return thunk


def evaluate(model, vec):
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
            x = np.zeros((1, MAXLEN, len(vec.chars)))
            for t, char in enumerate(sentence):
                x[0, t, vec.char_indices[char]] = 1.

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
        self.loss_ax.set_ylim(0.0, 3.0)
        self.loss_ax.set_xlim(0, self.maxt)

    def update(self, data):
        print('Graph update!')
        y, mat = data
        self.ydata.append(y)
        self.tdata = range(len(self.ydata))
        self.line.set_data(self.tdata, self.ydata)
        plot_matrix(mat, axes=self.mat_ax, show=False,
                    xlabels=self.xlabels, ylabels=self.ylabels)
        return self.line, self.mat_ax


def train_and_evaluate(path, inputs, outputs):
    X, y, vec = data_to_arrays(path)

    cell_size = 20  # 512
    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(cell_size, return_sequences=True, input_shape=(MAXLEN, len(vec.chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(vec.chars)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    fig, (top, bottom) = plt.subplots(nrows=2)
    graph = DynamicGraph(top, bottom, maxt=NUM_ITERS, xlabels=outputs + ['<unk>'], ylabels=inputs)

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
    for labels, path in zip([S_DATA, L_DATA], ['toy_data_s.txt', 'toy_data_l.txt']):
        inputs, outputs = labels
        train_and_evaluate(path, inputs, outputs)
