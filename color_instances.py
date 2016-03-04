from collections import namedtuple

try:
    from rugstk.data.munroecorpus import munroecorpus
except ImportError:
    import sys
    sys.stderr.write('Munroe corpus not found (have you run '
                     './dependencies?)\n')
    raise

from stanza.unstable.instance import Instance
from stanza.unstable.rng import get_rng


rng = get_rng()


def load_colors(h, s, v):
    return zip(*(munroecorpus.open_datafile(d) for d in [h, s, v]))


def get_training_instances(listener=False):
    h, s, v = munroecorpus.get_training_handles()
    insts = [
        (Instance(input=name,
                  output=color)
         if listener else
         Instance(input=color,
                  output=name))
        for name in h
        for color in load_colors(h[name], s[name], v[name])
    ]
    rng.shuffle(insts)
    return insts


def get_dev_instances(listener=False):
    handles = munroecorpus.get_dev_handles()
    insts = [
        (Instance(input=name,
                  output=tuple(color))
         if listener else
         Instance(input=tuple(color),
                  output=name))
        for name, handle in handles.iteritems()
        for color in munroecorpus.open_datafile(handle)
    ]
    rng.shuffle(insts)
    return insts


def tune_train(listener=False):
    all_train = get_training_instances(listener=listener)
    return all_train[:-100000]


def tune_eval(listener=False):
    all_train = get_training_instances(listener=listener)
    return all_train[-100000:]


def pairs_to_insts(data, listener=False):
    return [
        (Instance(input=name, output=color)
         if listener else
         Instance(input=color, output=name))
        for name, color in data
    ]


def empty_str(listener=False):
    data = [('', (167.74193548404, 96.3730569948, 75.6862745098))]
    return pairs_to_insts(data, listener=listener)


def one_word(listener=False):
    data = [('green', (167.74193548404, 96.3730569948, 75.6862745098))]
    return pairs_to_insts(data, listener=listener)


def scalar_imp_train(listener=False):
    data = [
        ('blue', (240., 100., 100.)),
        ('blue', (180., 100., 100.)),
        ('teal', (180., 100., 100.)),
    ]
    return pairs_to_insts(data, listener=listener)


def scalar_imp_test(listener=False):
    data = [
        ('blue', (240., 100., 100.)),
        ('blue', (180., 100., 100.)),
        ('teal', (240., 100., 100.)),
        ('teal', (180., 100., 100.)),
    ]
    return pairs_to_insts(data, listener=listener)


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'dev': DataSource(get_training_instances, get_dev_instances),
    'tune': DataSource(tune_train, tune_eval),
    '1word': DataSource(one_word, one_word),
    '0word': DataSource(empty_str, empty_str),
    'scalar': DataSource(scalar_imp_train, scalar_imp_test),
}
