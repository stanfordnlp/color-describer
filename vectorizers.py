import colorsys
import numpy as np
import operator
import theano.tensor as T
from collections import Sequence
from lasagne.layers import InputLayer, EmbeddingLayer, NINLayer, reshape, dimshuffle
from matplotlib.colors import hsv_to_rgb

import learners
import neural
from stanza.unstable import config
from stanza.unstable.rng import get_rng

rng = get_rng()


class SymbolVectorizer(object):
    '''
    Maps symbols from an alphabet/vocabulary of indefinite size to and from
    sequential integer ids.

    >>> vec = SymbolVectorizer()
    >>> vec.add_all(['larry', 'moe', 'larry', 'curly', 'moe'])
    >>> vec.vectorize_all(['curly', 'larry', 'moe', 'pikachu'])
    array([3, 1, 2, 0], dtype=int32)
    >>> vec.unvectorize_all([3, 3, 2])
    ['curly', 'curly', 'moe']
    '''
    def __init__(self):
        self.tokens = []
        self.token_indices = {}
        self.indices_token = {}
        self.add('<unk>')

    @property
    def num_types(self):
        return len(self.tokens)

    def add_all(self, symbols):
        for sym in symbols:
            self.add(sym)

    def add(self, symbol):
        if symbol not in self.token_indices:
            self.token_indices[symbol] = len(self.tokens)
            self.indices_token[len(self.tokens)] = symbol
            self.tokens.append(symbol)

    def vectorize(self, symbol):
        return (self.token_indices[symbol] if symbol in self.token_indices
                else self.token_indices['<unk>'])

    def vectorize_all(self, symbols):
        return np.array([self.vectorize(sym) for sym in symbols], dtype=np.int32)

    def unvectorize(self, index):
        return self.indices_token[index]

    def unvectorize_all(self, array):
        if hasattr(array, 'tolist'):
            array = array.tolist()
        return [self.unvectorize(elem) for elem in array]


class SequenceVectorizer(object):
    '''
    Maps sequences of symbols from an alphabet/vocabulary of indefinite size
    to and from sequential integer ids.

    >>> vec = SequenceVectorizer()
    >>> vec.add_all([['the', 'flat', 'cat', '</s>', '</s>'], ['the', 'cat', 'in', 'the', 'hat']])
    >>> vec.vectorize_all([['in', 'the', 'cat', 'flat', '</s>'],
    ...                    ['the', 'cat', 'sat', '</s>', '</s>']])
    array([[5, 1, 3, 2, 4],
           [1, 3, 0, 4, 4]], dtype=int32)
    >>> vec.unvectorize_all([[1, 3, 0, 5, 1], [1, 2, 3, 6, 4]])
    [['the', 'cat', '<unk>', 'in', 'the'], ['the', 'flat', 'cat', 'hat', '</s>']]
    '''
    def __init__(self):
        self.tokens = []
        self.token_indices = {}
        self.indices_token = {}
        self.max_len = 0
        self.add(['<unk>'])

    def add_all(self, sequences):
        for seq in sequences:
            self.add(seq)

    def add(self, sequence):
        self.max_len = max(self.max_len, len(sequence))
        for token in sequence:
            if token not in self.token_indices:
                self.token_indices[token] = len(self.tokens)
                self.indices_token[len(self.tokens)] = token
                self.tokens.append(token)

    def vectorize(self, sequence):
        return np.array([(self.token_indices[token] if token in self.token_indices
                          else self.token_indices['<unk>'])
                         for token in sequence], dtype=np.int32)

    def vectorize_all(self, sequences):
        return np.array([self.vectorize(seq) for seq in sequences], dtype=np.int32)

    def unvectorize(self, array):
        if hasattr(array, 'tolist'):
            array = array.tolist()
        return [(self.unvectorize(elem) if isinstance(elem, Sequence)
                 else self.indices_token[elem])
                for elem in array]

    def unvectorize_all(self, sequences):
        # unvectorize already accepts sequences of sequences.
        return self.unvectorize(sequences)


RANGES_RGB = (256.0, 256.0, 256.0)
RANGES_HSV = (360.0, 101.0, 101.0)
C_EPSILON = 1e-4


class ColorVectorizer(object):
    def vectorize_all(self, colors, hsv=None):
        '''
        :param colors: A sequence of length-3 vectors or 1D array-like objects containing
                      RGB coordinates in the range [0, 256).
        :param bool hsv: If `True`, input is assumed to be in HSV space in the range
                         [0, 360], [0, 100], [0, 100]; if `False`, input should be in RGB
                         space in the range [0, 256). `None` (default) means take the
                         color space from the value given to the constructor.
        :return np.ndarray: An array of the vectorized form of each color in `colors`
                            (first dimension is the index of the color in the `colors`).

        >>> BucketsVectorizer((2, 2, 2)).vectorize_all([(0, 0, 0), (255, 0, 0)])
        array([0, 4], dtype=int32)
        '''
        return np.array([self.vectorize(c, hsv=hsv) for c in colors])

    def unvectorize_all(self, colors, random=False, hsv=None):
        '''
        :param Sequence colors: An array or sequence of vectorized colors
        :param random: If true, sample a random color from each bucket. Otherwise,
                       return the center of the bucket. Some vectorizers map colors
                       one-to-one to vectorized versions; these vectorizers will
                       ignore the `random` argument.
        :param hsv: If `True`, return colors in HSV format; otherwise, RGB.
                    `None` (default) means take the color space from the value
                    given to the constructor.
        :return list(tuple(int)): The unvectorized version of each color in `colors`

        >>> BucketsVectorizer((2, 2, 2)).unvectorize_all([0, 4])
        [(64, 64, 64), (192, 64, 64)]
        >>> BucketsVectorizer((2, 2, 2)).unvectorize_all([0, 4], hsv=True)
        [(0, 0, 25), (0, 67, 75)]
        '''
        return [self.unvectorize(c, random=random, hsv=hsv) for c in colors]

    def visualize_distribution(self, dist):
        '''
        :param dist: A distribution over the buckets defined by this vectorizer
        :type dist: array-like with shape `(self.num_types,)``
        :return images: `list(`3-D `np.array` with `shape[2] == 3)`, three images
            with the last dimension being the channels (RGB) of cross-sections
            along each axis, showing the strength of the distribution as the
            intensity of the channel perpendicular to the cross-section.
        '''
        raise NotImplementedError

    def get_input_vars(self, id=None, recurrent=False):
        '''
        :param id: The string tag to use as a prefix in the variable names.
            If `None`, no prefix will be added. (Passing an empty string will
            result in adding a bare `'/'`, which is legal but probably not what
            you want.)
        :type id: str or None
        :param bool recurrent: If `True`, return input variables reflecting
            copying the input `k` times, where `k` is the recurrent sequence
            length. This means the input variables will have one more dimension
            than they would if they were input to a simple feed-forward layer.
        :return list(T.TensorVariable): The variables that should feed into the
            color component of the input layer of a neural network using this
            vectorizer.
        '''
        id_tag = (id + '/') if id else ''
        return [(T.imatrix if recurrent else T.ivector)(id_tag + 'colors')]

    def get_input_layer(self, input_vars, recurrent_length=0, cell_size=20, id=None):
        '''
        :param input_vars: The input variables returned from
            `get_input_vars`.
        :type input_vars: list(T.TensorVariable)
        :param recurrent_length: The number of steps to copy color representations
            for input to a recurrent unit. If `None`, allow variable lengths; if 0,
            produce output for a non-recurrent layer (this will create an input layer
            producing a tensor of rank one lower than the recurrent version).
        :type recurrent_length: int or None
        :param int cell_size: The number of dimensions of the final color representation.
        :param id: The string tag to use as a prefix in the layer names.
            If `None`, no prefix will be added. (Passing an empty string will
            result in adding a bare `'/'`, which is legal but probably not what
            you want.)
        :return Lasagne.Layer, list(Lasagne.Layer): The layer producing the color
            representation, and the list of input layers corresponding to each of
            the input variables (in the same order).
        '''
        raise NotImplementedError(self.get_input_layer)


class BucketsVectorizer(ColorVectorizer):
    '''
    Maps colors to a uniform grid of buckets.
    '''
    def __init__(self, resolution, hsv=False):
        '''
        :param resolution: A length-1 or length-3 sequence giving numbers of buckets
                           along each dimension of the RGB/HSV grid. If length-1, all
                           three dimensions will use the same number of buckets.
        :param bool hsv: If `True`, buckets will be laid out in a grid in HSV space;
                         otherwise, the grid will be in RGB space. Input and output
                         color spaces can be configured on a per-call basis by
                         using the `hsv` parameter of `vectorize` and `unvectorize`.
        '''
        if len(resolution) == 1:
            resolution = resolution * 3
        self.resolution = resolution
        self.num_types = reduce(operator.mul, resolution)
        self.hsv = hsv
        ranges = RANGES_HSV if hsv else RANGES_RGB
        self.bucket_sizes = tuple(d / r for d, r in zip(ranges, resolution))

    def vectorize(self, color, hsv=None):
        '''
        :param color: An length-3 vector or 1D array-like object containing
                      color coordinates.
        :param bool hsv: If `True`, input is assumed to be in HSV space in the range
                         [0, 360], [0, 100], [0, 100]; if `False`, input should be in RGB
                         space in the range [0, 256). `None` (default) means take the
                         color space from the value given to the constructor.
        :return int: The bucket id for `color`

        >>> BucketsVectorizer((2, 2, 2)).vectorize((0, 0, 0))
        0
        >>> BucketsVectorizer((2, 2, 2)).vectorize((255, 0, 0))
        4
        >>> BucketsVectorizer((2, 2, 2)).vectorize((240, 100, 100), hsv=True)
        ... # HSV (240, 100, 100) = RGB (0, 0, 255)
        1
        >>> BucketsVectorizer((2, 2, 2), hsv=True).vectorize((0, 0, 0))
        0
        >>> BucketsVectorizer((2, 2, 2), hsv=True).vectorize((240, 0, 0))
        ... # yes, this is also black. Using HSV buckets is a questionable decision.
        4
        >>> BucketsVectorizer((2, 2, 2), hsv=True).vectorize((0, 255, 0), hsv=False)
        ... # RGB (0, 255, 0) = HSV (120, 100, 100)
        3
        '''
        if hsv is None:
            hsv = self.hsv

        if hsv and not self.hsv:
            c_hsv = color
            c_rgb_0_1 = colorsys.hsv_to_rgb(*(d * 1.0 / r for d, r in zip(c_hsv, RANGES_HSV)))
            color_internal = tuple(int(d * 255.99) for d in c_rgb_0_1)
        elif not hsv and self.hsv:
            c_rgb = color
            c_hsv_0_1 = colorsys.rgb_to_hsv(*(d / 256.0 for d in c_rgb))
            color_internal = tuple(int(d * (r - C_EPSILON)) for d, r in zip(c_hsv_0_1, RANGES_HSV))
        else:
            ranges = RANGES_HSV if self.hsv else RANGES_RGB
            color_internal = tuple(min(d, r - C_EPSILON) for d, r in zip(color, ranges))

        bucket_dims = [int(e // r) for e, r in zip(color_internal, self.bucket_sizes)]
        result = (bucket_dims[0] * self.resolution[1] * self.resolution[2] +
                  bucket_dims[1] * self.resolution[2] +
                  bucket_dims[2])
        assert (0 <= result < self.num_types), (color, result)
        return np.int32(result)

    def unvectorize(self, bucket, random=False, hsv=None):
        '''
        :param int bucket: The id of a color bucket
        :param random: If `True`, sample a random color from the bucket. Otherwise,
                       return the center of the bucket.
        :param hsv: If `True`, return colors in HSV format [0 <= hue <= 360,
                    0 <= sat <= 100, 0 <= val <= 100]; if `False`, RGB
                    [0 <= r/g/b <= 256]. `None` (default) means take the
                    color space from the value given to the constructor.
        :return tuple(int): A color from the bucket with id `bucket`.

        >>> BucketsVectorizer((2, 2, 2)).unvectorize(0)
        (64, 64, 64)
        >>> BucketsVectorizer((2, 2, 2)).unvectorize(4)
        (192, 64, 64)
        >>> BucketsVectorizer((2, 2, 2)).unvectorize(4, hsv=True)
        (0, 67, 75)
        >>> BucketsVectorizer((2, 2, 2), hsv=True).unvectorize(0)
        (90, 25, 25)
        >>> BucketsVectorizer((2, 2, 2), hsv=True).unvectorize(4)
        (270, 25, 25)
        >>> BucketsVectorizer((2, 2, 2), hsv=True).unvectorize(4, hsv=False)
        (55, 47, 63)
        '''
        if hsv is None:
            hsv = self.hsv
        bucket_start = (
            (bucket / (self.resolution[1] * self.resolution[2]) % self.resolution[0]),
            (bucket / self.resolution[2]) % self.resolution[1],
            bucket % self.resolution[2],
        )
        color = tuple((rng.randint(d * size, (d + 1) * size) if random
                       else (d * size + size // 2))
                      for d, size in zip(bucket_start, self.bucket_sizes))
        if self.hsv:
            c_hsv = tuple(int(d) for d in color)
            c_rgb_0_1 = colorsys.hsv_to_rgb(*(d * 1.0 / r for d, r in zip(color, RANGES_HSV)))
            c_rgb = tuple(int(d * 256.0) for d in c_rgb_0_1)
        else:
            c_rgb = tuple(int(d) for d in color)
            c_hsv_0_1 = colorsys.rgb_to_hsv(*(d / 256.0 for d in color))
            c_hsv = tuple(int(d * r) for d, r in zip(c_hsv_0_1, RANGES_HSV))

        if hsv:
            return c_hsv
        else:
            return c_rgb

    def visualize_distribution(self, dist):
        '''
        >>> BucketsVectorizer((2, 2, 2)).visualize_distribution([0, 0.25, 0, 0.5,
        ...                                                    0, 0, 0, 0.25])
        ... # doctest: +NORMALIZE_WHITESPACE
        [array([[[  0,  64,  64], [ 85,  64, 192]],
                [[  0, 192,  64], [255, 192, 192]]]),
         array([[[ 64,   0,  64], [ 64, 255, 192]],
                [[192,   0,  64], [192,  85, 192]]]),
         array([[[ 64,  64, 127], [ 64, 192, 255]],
                [[192,  64,   0], [192, 192, 127]]])]
        '''
        dist_3d = np.asarray(dist).reshape(self.resolution)
        # Compute background: RGB/HSV for each bucket along each face with one channel set to 0
        x, y, z = self.bucket_sizes
        ranges = RANGES_HSV if self.hsv else RANGES_RGB
        rx, ry, rz = ranges
        images = [
            np.array(
                np.meshgrid(0, np.arange(y // 2, ry, y), np.arange(z // 2, rz, z))
            ).squeeze(2).transpose((1, 2, 0)).astype(np.int),
            np.array(
                np.meshgrid(np.arange(x // 2, rx, x), 0, np.arange(z // 2, rz, z))
            ).squeeze(1).transpose((1, 2, 0)).astype(np.int),
            np.array(
                np.meshgrid(np.arange(x // 2, rx, x), np.arange(y // 2, ry, y), 0)
            ).squeeze(3).transpose((2, 1, 0)).astype(np.int),
        ]
        for axis in range(3):
            xsection = dist_3d.sum(axis=axis)
            xsection /= xsection.max()
            if self.hsv:
                im_float = images[axis].astype(np.float) / np.array(RANGES_HSV)
                im_float[:, :, axis] = xsection
                images[axis] = (hsv_to_rgb(im_float) *
                                (np.array(RANGES_RGB) - C_EPSILON)).astype(np.int)
            else:
                images[axis][:, :, axis] = (xsection * (ranges[axis] - C_EPSILON)).astype(np.int)
        return images

    def get_input_layer(self, input_vars, recurrent_length=0, cell_size=20, id=None):
        options = config.options()
        id_tag = (id + '/') if id else ''
        (input_var,) = input_vars
        shape = (None,) if recurrent_length == 0 else (None, recurrent_length)
        l_color = InputLayer(shape=shape, input_var=input_var,
                             name=id_tag + 'color_input')
        l_color_embed = EmbeddingLayer(l_color, input_size=self.num_types,
                                       output_size=cell_size,
                                       name=id_tag + 'color_embed')
        l_hidden_color = (l_color_embed
                          if recurrent_length == 0 else
                          dimshuffle(l_color_embed, (0, 2, 1)))
        NL = neural.NONLINEARITIES
        for i in range(1, options.speaker_hidden_color_layers + 1):
            l_hidden_color = NINLayer(l_hidden_color, num_units=options.speaker_cell_size,
                                      nonlinearity=NL[options.speaker_nonlinearity],
                                      name=id_tag + 'hidden_color%d' % i)
        l_hidden_color = (l_hidden_color
                          if recurrent_length == 0 else
                          dimshuffle(l_hidden_color, (0, 2, 1)))
        return l_hidden_color, [l_color]


class MSVectorizer(ColorVectorizer):
    '''
    Maps colors to several overlaid uniform grid of buckets with
    different resolutions, and concatenates these representations.
    '''
    def __init__(self, resolution='ignored', hsv='ignored'):
        self.num_types = np.prod(learners.HistogramLearner.GRANULARITY[0])
        self.buckets = [BucketsVectorizer(res, hsv=True)
                        for res in learners.HistogramLearner.GRANULARITY]

    def vectorize(self, color, hsv=None):
        '''
        :param color: An length-3 vector or 1D array-like object containing
                      color coordinates.
        :param bool hsv: If `True`, input is assumed to be in HSV space in the range
                         [0, 360], [0, 100], [0, 100]; if `False`, input should be in RGB
                         space in the range [0, 256). `None` (default) means take the
                         color space from the value given to the constructor.
        :return int: The bucket id for `color`

        >>> MSVectorizer().vectorize((255, 0, 0), hsv=False)
        ... # RGB (0, 0, 255) = HSV (0, 100, 100)
        array([   99,  9024, 10125], dtype=int32)
        >>> MSVectorizer().vectorize((240, 100, 100))
        array([ 6099,  9774, 10125], dtype=int32)
        '''
        buckets = np.array([b.vectorize(color, hsv=hsv) for b in self.buckets], dtype=np.int32)
        prev = np.array([0] + [b.num_types for b in self.buckets[:-1]])
        return buckets + np.cumsum(prev, dtype=np.int32)

    def unvectorize(self, bucket, random=False, hsv=None):
        '''
        :param int bucket: The ids of the color buckets for each resolution
        :param random: If `True`, sample a random color from the bucket. Otherwise,
                       return the center of the bucket.
        :param hsv: If `True`, return colors in HSV format [0 <= hue <= 360,
                    0 <= sat <= 100, 0 <= val <= 100]; if `False`, RGB
                    [0 <= r/g/b <= 256]. `None` (default) means take the
                    color space from the value given to the constructor.
        :return tuple(int): A color from the bucket with ids `bucket`. Note that
                            only the id from the highest-resolution grid is
                            used to identify the bucket (the others are redundant).

        >>> MSVectorizer().unvectorize([99, 9024, 10125], hsv=False)
        (243, 19, 12)
        >>> MSVectorizer().unvectorize([6099, 9774, 10125])
        (242, 95, 95)
        '''
        return self.buckets[0].unvectorize(bucket[0], random=random, hsv=hsv)

    def visualize_distribution(self, dist):
        return self.buckets[0].visualize_distribution(dist)

    def get_input_vars(self, id=None, recurrent=False):
        id_tag = (id + '/') if id else ''
        return [(T.itensor3 if recurrent else T.imatrix)(id_tag + 'colors')]

    def get_input_layer(self, input_vars, recurrent_length=0, cell_size=20, id=None):
        options = config.options()
        id_tag = (id + '/') if id else ''
        (input_var,) = input_vars
        shape = ((None, len(self.buckets))
                 if recurrent_length == 0 else
                 (None, recurrent_length, len(self.buckets)))
        l_color = InputLayer(shape=shape, input_var=input_var,
                             name=id_tag + 'color_input')
        l_color_embed = EmbeddingLayer(l_color, input_size=sum(b.num_types for b in self.buckets),
                                       output_size=cell_size,
                                       name=id_tag + 'color_embed')

        dims = (([0], -1) if recurrent_length == 0 else ([0], [1], -1))
        l_color_flattened = reshape(l_color_embed, dims)

        l_hidden_color = (l_color_flattened
                          if recurrent_length == 0 else
                          dimshuffle(l_color_flattened, (0, 2, 1)))
        NL = neural.NONLINEARITIES
        for i in range(1, options.speaker_hidden_color_layers + 1):
            l_hidden_color = NINLayer(l_hidden_color, num_units=options.speaker_cell_size,
                                      nonlinearity=NL[options.speaker_nonlinearity],
                                      name=id_tag + 'hidden_color%d' % i)
        l_hidden_color = (l_hidden_color
                          if recurrent_length == 0 else
                          dimshuffle(l_hidden_color, (0, 2, 1)))
        return l_hidden_color, [l_color]


class RawVectorizer(ColorVectorizer):
    '''
    Vectorizes colors with the identity function (each color is simply represented
    by its raw 3-dimensional vector, RGB or HSV).
    '''
    def __init__(self, resolution='ignored', hsv=False):
        '''
        :param bool hsv: If `True`, the internal representation used by the vectorizer
                         will be HSV. Input and output color spaces can be configured
                         on a per-call basis by using the `hsv` parameter of
                         `vectorize` and `unvectorize`.
        '''
        if hsv:
            resolution = (360, 101, 101)
        else:
            resolution = (256, 256, 256)
        self.num_types = reduce(operator.mul, resolution)
        self.hsv = hsv

    def vectorize(self, color, hsv=None):
        '''
        :param color: An length-3 vector or 1D array-like object containing
                      color coordinates.
        :param bool hsv: If `True`, input is assumed to be in HSV space in the range
                         [0, 359], [0, 100], [0, 100]; if `False`, input should be in RGB
                         space in the range [0, 255]. `None` (default) means take the
                         color space from the value given to the constructor.
        :return np.ndarray: The color in the internal representation of the vectorizer,
                            a vector of shape (3,). The values of this vector will be
                            scaled and shifted to lie in the range [-1, 1].

        >>> RawVectorizer().vectorize((255, 0, 0))
        array([ 1., -1., -1.])
        >>> RawVectorizer().vectorize((0, 100, 100), hsv=True)
        array([ 1., -1., -1.])
        >>> RawVectorizer(hsv=True).vectorize((0, 100, 100))
        array([-1.,  1.,  1.])
        >>> RawVectorizer(hsv=True).vectorize((255, 0, 0), hsv=False)
        array([-1.,  1.,  1.])
        '''
        if hsv is None:
            hsv = self.hsv

        if hsv and not self.hsv:
            c_hsv = color
            color_0_1 = colorsys.hsv_to_rgb(*(d / (r - 1.0) for d, r in zip(c_hsv, RANGES_HSV)))
        elif not hsv and self.hsv:
            c_rgb = color
            color_0_1 = colorsys.rgb_to_hsv(*(d / (r - 1.0) for d, r in zip(c_rgb, RANGES_RGB)))
        else:
            ranges = RANGES_HSV if self.hsv else RANGES_RGB
            color_0_1 = tuple(d / (r - 1.0) for d, r in zip(color, ranges))
        color_internal = tuple(d * 2.0 - 1.0 for d in color_0_1)

        return np.array(color_internal)

    def unvectorize(self, color, random='ignored', hsv=None):
        '''
        :param np.ndarray color: A vectorized color in the internal color space
        :param hsv: If `True`, return colors in HSV format [0 <= hue <= 360,
                    0 <= sat <= 100, 0 <= val <= 100]; if `False`, RGB
                    [0 <= r/g/b <= 256]. `None` (default) means take the
                    color space from the value given to the constructor.
        :return tuple(int): The color in the requested output space,
                            in the range [0, 255] for RGB and
                            [0, 359], [0, 100], [0, 100] for HSV.

        >>> RawVectorizer().unvectorize((1., -1., -1.))
        (255, 0, 0)
        >>> RawVectorizer().unvectorize((1., -1., -1.), hsv=True)
        (0, 100, 100)
        >>> RawVectorizer(hsv=True).unvectorize((-1., 1., 1.))
        (0, 100, 100)
        >>> RawVectorizer(hsv=True).unvectorize((-1., 1., 1.), hsv=False)
        (255, 0, 0)
        '''
        if hsv is None:
            hsv = self.hsv
        color_0_1 = tuple((d + 1.0) / 2.0 for d in color)
        if self.hsv:
            c_hsv = tuple(int(d * (r - C_EPSILON)) for d, r in zip(color_0_1, RANGES_HSV))
            c_rgb_0_1 = colorsys.hsv_to_rgb(*(d for d in color_0_1))
            c_rgb = tuple(int(d * (r - C_EPSILON)) for d, r in zip(c_rgb_0_1, RANGES_RGB))
        else:
            c_rgb = tuple(int(d * (r - C_EPSILON)) for d, r in zip(color_0_1, RANGES_RGB))
            c_hsv_0_1 = colorsys.rgb_to_hsv(*(d for d in color_0_1))
            c_hsv = tuple(int(d * (r - C_EPSILON)) for d, r in zip(c_hsv_0_1, RANGES_HSV))

        if hsv:
            return c_hsv
        else:
            return c_rgb

    def get_input_vars(self, id=None, recurrent=False):
        id_tag = (id + '/') if id else ''
        return [(T.tensor3 if recurrent else T.matrix)(id_tag + 'colors')]

    def get_input_layer(self, input_vars, recurrent_length=0, cell_size=20, id=None):
        options = config.options()
        id_tag = (id + '/') if id else ''
        (input_var,) = input_vars
        shape = (None, 3) if recurrent_length == 0 else (None, recurrent_length, 3)
        l_color = InputLayer(shape=shape, input_var=input_var,
                             name=id_tag + 'color_input')
        l_hidden_color = l_color
        NL = neural.NONLINEARITIES
        for i in range(1, options.speaker_hidden_color_layers + 1):
            l_hidden_color = NINLayer(l_hidden_color, num_units=cell_size,
                                      nonlinearity=NL[options.speaker_nonlinearity],
                                      name=id_tag + 'hidden_color%d' % i)
        return l_hidden_color, [l_color]
