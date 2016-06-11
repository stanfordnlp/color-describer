import cPickle as pickle
import theano  # NOQA: for doctest (get that dot_parser warning out of the way with)

from stanza.research.instance import Instance
from colorutils import rgb_to_hsv, hsl_to_hsv


class ColorDescriber(object):
    '''
    A wrapper class for the LSTM color-description model described in
    "Learning to Generate Compositional Color Descriptions".
    '''
    def __init__(self, picklefile=None):
        '''
        :param file picklefile: An open file-like object from which to
            load the model. Can be produced either from a normal experiment
            run or a quickpickle.py run. If `None`, try to load the default
            quickpickle file (this is less future-proof than the normal
            experiment-produced pickle files).
        '''
        if picklefile is None:
            with open('models/lstm_fourier_quick.p', 'rb') as infile:
                self.model = pickle.load(infile)
        else:
            self.model = pickle.load(picklefile)
        self.model.options.verbosity = 0

    def describe(self, color, format='rgb', sample=False):
        '''
        Return a description for `color`, which is expressed in the colorspace
        given by `format` (one of 'rgb', 'hsv', 'hsl'). If `sample` is `True`,
        return a random description sampled from the model's probability
        distribution; otherwise return the most likely, common description.

        >>> cd = ColorDescriber()  # doctest: +ELLIPSIS
        >>> cd.describe((0, 0, 255))
        'blue'
        >>> cd.describe((0, 100, 100), format='hsv')
        'red'
        '''
        return self.describe_all([color], format=format, sample=sample)[0]

    def describe_all(self, colors, format='rgb', sample=False):
        '''
        Return a list of descriptions, one for each color in `colors`, which
        is expressed in the colorspace given by `format`
        (one of 'rgb', 'hsv', 'hsl'). If `sample` is `True`,
        return descriptions sampled from the model's probability
        distribution; otherwise return the most likely, common descriptions.

        >>> cd = ColorDescriber()
        >>> cd.describe_all([(255, 0, 0), (0, 0, 255)])
        ['red', 'blue']
        '''
        convert = {
            'hsv': (lambda c: c),
            'hsl': hsl_to_hsv,
            'rgb': rgb_to_hsv,
        }[format]
        insts = [Instance(convert(c)) for c in colors]
        return self.model.predict(insts, random=sample)

    def score(self, color, description, format='rgb'):
        '''
        Return the log probability (base e) of `description` given `color`,
        where `color` is expressed in the colorspace given by `format`
        (one of 'rgb', 'hsv', 'hsl').

        >>> cd = ColorDescriber()
        >>> cd.score((0, 0, 255), 'blue')  # doctest: +ELLIPSIS
        -0.26...
        '''
        return self.score_all([color], [description], format=format)[0]

    def score_all(self, colors, descriptions, format='rgb'):
        '''
        Return a list of log probabilities (base e) for the descriptions
        in `descriptions`, conditioned on the corresponding colors in `colors`.
        `descriptions` and `colors` have the same length.
        is expressed in the colorspace given by `format`
        (one of 'rgb', 'hsv', 'hsl'). If `sample` is `True`,
        return descriptions sampled from the model's probability
        distribution; otherwise return the most likely, common descriptions.

        >>> cd = ColorDescriber()
        >>> cd.score_all([(255, 0, 0), (0, 0, 255)], ['red', 'blue'])  # doctest: +ELLIPSIS
        [-0.23..., -0.26...]
        '''
        convert = {
            'hsv': (lambda c: c),
            'hsl': hsl_to_hsv,
            'rgb': rgb_to_hsv,
        }[format]
        insts = [Instance(convert(c), d) for c, d in zip(colors, descriptions)]
        return self.model.score(insts)
