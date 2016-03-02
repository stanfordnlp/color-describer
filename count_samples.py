import sys
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def report_samples(infile):
    counts = get_sample_counts(infile)
    for agent in sorted(counts.keys()):
        print('Agent: %s' % agent)
        print('Types')
        agent_types = sorted(set(c for epoch in counts[agent] for c in epoch))
        print('Counters')
        counters = [Counter(epoch) for epoch in counts[agent]]
        print('Counts')
        mat = np.array([[c[t] for c in counters] for t in agent_types])
        print('Plots')
        for t, row in zip(agent_types, range(mat.shape[0])):
            print('Plot: %s' % t)
            plt.plot(np.arange(len(counters)), mat[row, :])
        plt.legend(['%s: %s' % (agent, t) for t in agent_types])
        plt.show()


def get_sample_counts(infile):
    agent = None
    counts = defaultdict(list)
    current_samples = []
    for i, line in enumerate(infile):
        if i % 100000 == 0:
            print('Line %d' % i)
        line = line.strip()
        if agent is None:
            if line.endswith(' samples:'):
                agent = line[:-len(' samples:')]
        else:
            if ' -> ' in line:
                current_samples.append(parse_sample(line))
            else:
                counts[agent].append(current_samples)
                current_samples = []
                if line.endswith(' samples:'):
                    agent = line[:-len(' samples:')]
                else:
                    agent = None
    return counts


def parse_sample(line):
    '''
    >>> parse_sample("'teal' -> (180, 100, 100)")
    "'teal' -> +G"
    >>> parse_sample("(240, 100, 100) -> 'blue'")
    "-G -> 'blue'"
    '''
    inp, out = line.split(' -> ')
    return '%s -> %s' % (normalize_color(inp), normalize_color(out))


def normalize_color(fragment):
    '''
    >>> normalize_color("'blue'")
    "'blue'"
    >>> normalize_color("(0, 100, 100)")
    '-G'
    >>> normalize_color("(120, 100, 100)")
    '+G'
    >>> normalize_color("(180, 100, 100)")
    '+G'
    '''
    value = eval(fragment)
    if isinstance(value, tuple):
        hsv_0_1 = (value[0] / 360.0, value[1] / 100.0, value[2] / 100.0)
        r, g, b = colorsys.hsv_to_rgb(*hsv_0_1)
        return '+G' if g > 0.5 else '-G'
    else:
        return fragment


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as infile:
        report_samples(infile)
