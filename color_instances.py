import colorsys
from collections import namedtuple

try:
    from rugstk.data.munroecorpus import munroecorpus
except ImportError:
    import sys
    sys.stderr.write('Munroe corpus not found (have you run '
                     './dependencies?)\n')
    raise

from stanza.unstable import config
from stanza.unstable.instance import Instance
from stanza.unstable.rng import get_rng


rng = get_rng()

parser = config.get_options_parser()
parser.add_argument('--num_distractors', type=int, default=4,
                    help='The number of random colors to include in addition to the true '
                         'color in generating reference game instances. Ignored if not '
                         'using one of the `ref_` data sources.')


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


def two_word(listener=False):
    data = [('shocking green', (167.74193548404, 96.3730569948, 75.6862745098))]
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


def scalar_imp_level2_train(listener=False):
    data = [
        ('blue', (240., 100., 100.)),
        ('blue', (170., 100., 70.)),
        ('green', (170., 100., 70.)),
        ('green', (80., 100., 100.)),
        ('yellow', (80., 100., 100.)),
    ]
    return pairs_to_insts(data, listener=listener)


def scalar_imp_level2_test(listener=False):
    data = [
        ('blue', (240., 100., 100.)),
        ('blue', (170., 100., 70.)),
        ('blue', (80., 100., 100.)),
        ('green', (240., 100., 100.)),
        ('green', (170., 100., 70.)),
        ('green', (80., 100., 100.)),
        ('yellow', (240., 100., 100.)),
        ('yellow', (170., 100., 70.)),
        ('yellow', (80., 100., 100.)),
    ]
    return pairs_to_insts(data, listener=listener)


def reference_game_train(gen_func):
    def generate_refgame_train(listener=False):
        return reference_game(get_training_instances(), gen_func, listener=listener)
    return generate_refgame_train


def reference_game_test(gen_func):
    def generate_refgame_test(listener=False):
        return reference_game(get_training_instances(), gen_func, listener=listener)
    return generate_refgame_test


def reference_game(insts, gen_func, listener=False):
    options = config.options()
    result = []
    for inst in insts:
        color = inst.output if listener else inst.input
        distractors = [gen_func(color) for _ in range(options.num_distractors)]
        answer = rng.randint(0, len(distractors) + 1)
        context = distractors[:answer] + [color] + distractors[answer:]
        ref_inst = (Instance(inst.input, answer, alt_outputs=context)
                    if listener else
                    Instance(answer, inst.output, alt_inputs=context))
        result.append(ref_inst)
    return result


def uniform(color):
    r, g, b = rng.uniform(size=(3,))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s * 100.0, v * 100.0


def linear_rgb(color):
    coord = rng.randint(0, 3)
    val = rng.uniform()
    h, s, v = color
    result = list(colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0))
    result[coord] = val
    h, s, v = colorsys.rgb_to_hsv(*result)
    return h * 360.0, s * 100.0, v * 100.0


def linear_hsv(color):
    coord = rng.randint(0, 3)
    val = rng.uniform()
    h, s, v = color
    result = [h / 360.0, s / 100.0, v / 100.0]
    result[coord] = val
    h, s, v = result
    return h * 360.0, s * 100.0, v * 100.0


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'dev': DataSource(get_training_instances, get_dev_instances),
    'tune': DataSource(tune_train, tune_eval),
    '2word': DataSource(two_word, two_word),
    '1word': DataSource(one_word, one_word),
    '0word': DataSource(empty_str, empty_str),
    'scalar': DataSource(scalar_imp_train, scalar_imp_test),
    'scalar_lv2': DataSource(scalar_imp_level2_train, scalar_imp_level2_test),
    'ref_uni': DataSource(reference_game_train(uniform), reference_game_test(uniform)),
    'ref_linrgb': DataSource(reference_game_train(linear_rgb), reference_game_test(linear_rgb)),
    'ref_linhsv': DataSource(reference_game_train(linear_hsv), reference_game_test(linear_hsv)),
}
