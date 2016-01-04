# Hybrid of Lasagne MNIST example
#   <https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py>
# and Keras LSTM example
#   <https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py>
# modified to apply to color description understanding.
from __future__ import print_function
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.layers import ReshapeLayer, NonlinearityLayer, ConcatLayer
from lasagne.layers import get_output, get_all_params
from lasagne.layers.recurrent import LSTMLayer
from lasagne.objectives import squared_error, categorical_crossentropy
from lasagne.nonlinearities import softmax
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
from collections import namedtuple, Sequence
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
    tokens = ['<MASK>', '<s>', '</s>'] + list(set(t for color, desc in examples for t in desc))
    print('total tokens:', len(tokens))
    max_len = max(len(desc) for color, desc in examples) + 2
    print('max len (including <s> </s>):', max_len)
    token_indices = dict((c, i) for i, c in enumerate(tokens))
    indices_token = dict((i, c) for i, c in enumerate(tokens))

    # TODO: bucket colors
    return Vectorization(tokens, token_indices, indices_token, max_len)


def data_to_arrays_listener(path):
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
    X = np.zeros((len(sentences), vec.max_len, len(vec.tokens)), dtype=np.bool)
    y = np.zeros((len(sentences), 3), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        for t, token in enumerate(sentence):
            X[i, t, vec.token_indices[token]] = 1
        y[i, :] = colors[i]

    return X, y, vec


def data_to_arrays_speaker(path):
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
    c = np.zeros((len(colors), 3), dtype=np.float32)
    P = np.zeros((len(previous), vec.max_len - 1, len(vec.tokens)), dtype=np.float32)
    mask = np.zeros((len(previous), vec.max_len - 1), dtype=np.float32)
    N = np.zeros((len(next_tokens), vec.max_len - 1, len(vec.tokens)), dtype=np.float32)
    for i, (color, prev, next) in enumerate(zip(colors, previous, next_tokens)):
        c[i, :] = colors[i]
        for t, token in enumerate(prev):
            P[i, t, vec.token_indices[token]] = 1
        for t, token in enumerate(next):
            N[i, t, vec.token_indices[token]] = 1
            mask[i, t] = (token != '<MASK>')
    c = np.tile(c[:, np.newaxis, :], [1, vec.max_len - 1, 1])

    print('c:')
    print(c)
    print('P:')
    print(P)
    print('mask:')
    print(mask)
    print('N:')
    print(N)
    return (c, P, mask), N, vec


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


def get_log_prob_listener(model, input, output, vec, verbose=False):
    sentence = ['<s>'] * (vec.max_len - 1 - len(input)) + input + ['</s>']
    x = np.zeros((1, vec.max_len, len(vec.tokens)))
    for t, in_token in enumerate(sentence):
        x[0, t, vec.token_indices[in_token]] = 1.
    preds = model.predict(x, verbose=0)[0]
    if verbose:
        print('preds for %s: %s' % (input, preds))
    return scipy.stats.multivariate_normal(preds, STDEV).logpdf(output)


def get_log_prob_speaker(model, input, output, vec, verbose=False):
    if verbose:
        print('input = %s, output = %s' % (input, output))
    full = ['<s>'] + output + ['</s>'] + ['<MASK>'] * (vec.max_len - 2 - len(output))
    prev = full[:-1]
    next = full[1:]
    c = np.array([[input] * (vec.max_len - 1)])
    P = np.zeros((1, vec.max_len - 1, len(vec.tokens)))
    mask = np.zeros((1, vec.max_len - 1))
    for t, (in_token, out_token) in enumerate(zip(prev, next)):
        P[0, t, vec.token_indices[in_token]] = 1.
        mask[0, t] = (out_token != '<MASK>')
    preds = model.predict([c, P, mask], verbose=0)
    if verbose:
        print('preds for %s: %s' % (input, preds))
    log_prob = 1.0
    for t, out_token in enumerate(next):
        if out_token != '<MASK>':
            log_prob *= preds[0, t, vec.token_indices[out_token]]
    return log_prob


'''
S_DATA = [
    [(255, 0, 0), (0, 0, 255)],
    [['red'], ['green'], ['blue'],
     ['bright', 'red'], ['bright', 'green'], ['bright', 'blue']],
]
'''
S_DATA = [
    [(128, 0, 0), (255, 0, 0), (0, 0, 255), (64, 64, 255)],
    [['red'], ['bright', 'red'], ['blue'], ['bright', 'blue']],
]
L_DATA = [
    [['red'], ['bright', 'red'], ['blue'], ['bright', 'blue']],
    [(128, 0, 0), (255, 0, 0), (0, 0, 255), (64, 64, 255)],
]


def matrix(inputs, outputs, vec, get_log_prob):
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
        get_log_prob(model, inputs[0], outputs[0], vec, verbose=True)
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


def build_model_listener(vec):
    cell_size = 20  # 512
    input_var = T.tensor3('inputs')
    target_var = T.matrix('targets')

    l_in = InputLayer(shape=(None, vec.max_len, len(vec.tokens)), input_var=input_var)
    l_lstm1 = LSTMLayer(l_in, num_units=cell_size)
    l_lstm1_drop = DropoutLayer(l_lstm1, p=0.2)
    l_lstm2 = LSTMLayer(l_lstm1_drop, num_units=cell_size)
    l_lstm2_drop = DropoutLayer(l_lstm2, p=0.2)

    l_hidden = DenseLayer(l_lstm2_drop, num_units=cell_size, nonlinearity=None)
    l_hidden_drop = DropoutLayer(l_hidden, p=0.2)
    l_out = DenseLayer(l_hidden_drop, num_units=3, nonlinearity=None)

    return LasagneModel(input_var, target_var, l_out, loss=squared_error, optimizer=rmsprop)


def build_model_speaker(vec):
    cell_size = 20  # 512
    input_var = T.tensor3('inputs')
    prev_output_var = T.tensor3('previous')
    mask_var = T.matrix('mask')
    target_var = T.tensor3('targets')

    '''
    l_in = InputLayer(shape=(None, 3), input_var=input_var)
    l_hidden1 = DenseLayer(l_in, num_units=cell_size, nonlinearity=tanh)
    l_hidden1_drop = DropoutLayer(l_hidden1, p=0.2)
    l_hidden2 = DenseLayer(l_hidden1_drop, num_units=cell_size, nonlinearity=tanh)
    l_hidden2_drop = DropoutLayer(l_hidden2, p=0.2)
    '''

    l_color = InputLayer(shape=(None, vec.max_len - 1, 3), input_var=input_var)
    l_prev_out = InputLayer(shape=(None, vec.max_len - 1, len(vec.tokens)),
                            input_var=prev_output_var)
    l_in = ConcatLayer([l_color, l_prev_out], axis=2)
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
                        loss=categorical_crossentropy, optimizer=rmsprop)


def train_and_evaluate(path, inputs, outputs, listener=False):
    data_to_arrays = data_to_arrays_listener if listener else data_to_arrays_speaker
    Xs, y, vec = data_to_arrays(path)

    # build the model: 2 stacked LSTM
    print('Build model...')
    build_model = build_model_listener if listener else build_model_speaker
    model = build_model(vec)

    fig, (top, bottom) = plt.subplots(nrows=2)
    graph = DynamicGraph(top, bottom, maxt=NUM_ITERS, xlabels=outputs, ylabels=inputs)

    # pass a generator in "emitter" to produce data for the update func
    print('Build animation...')
    print('graph.update = %s' % repr(graph.update))
    get_log_prob = get_log_prob_listener if listener else get_log_prob_speaker
    emitter = lambda: train(model, Xs, y, matrix(inputs, outputs, vec, get_log_prob))
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
    def __init__(self, input_vars, target_var, l_out, loss, optimizer):
        if not isinstance(input_vars, Sequence):
            input_vars = [input_vars]

        prediction = get_output(l_out)
        test_prediction = get_output(l_out, deterministic=True)
        mean_loss = loss(prediction, target_var).mean()
        params = get_all_params(l_out, trainable=True)
        updates = optimizer(mean_loss, params, learning_rate=0.001)
        self.train_fn = theano.function(input_vars + [target_var], mean_loss, updates=updates)
        self.predict_fn = theano.function(input_vars, test_prediction)

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


def main():
    for labels, listener in [(S_DATA, False)]:  # [(L_DATA, True), (S_DATA, False)]:
        path = 'toy_data_color.txt'
        inputs, outputs = labels
        train_and_evaluate(path, inputs, outputs, listener=listener)


if __name__ == '__main__':
    main()
