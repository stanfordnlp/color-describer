import numpy as np

try:
    from rugstk.data.munroecorpus import munroecorpus
except ImportError:
    import sys
    sys.stderr.write('Munroe corpus not found (have you run '
                     './dependencies?)\n')
    raise

from bt.instance import Instance


random = np.random.RandomState(np.uint32(hash('reproducible')))


def load_colors(h, s, v):
    return zip(munroecorpus.open_datafile(d) for d in [h, s, v])


def get_training_instances():
    h, s, v = munroecorpus.get_training_handles()
    all_color_names = sorted(h.keys())
    insts = [
        Instance(input=color,
                 output=name,
                 alt_outputs=all_color_names)
        for name in h
        for color in load_colors(h[name], s[name], v[name])
    ]
    random.shuffle(insts)
    return insts


def get_dev_instances():
    handles = munroecorpus.get_dev_handles()
    all_color_names = sorted(handles.keys())
    insts = [
        Instance(input=tuple(color),
                 output=name,
                 alt_outputs=all_color_names)
        for name, handle in handles.iteritems()
        for color in munroecorpus.open_datafile(handle)
    ]
    random.shuffle(insts)
    return insts
