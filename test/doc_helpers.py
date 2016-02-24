'''
These doctests test the functionality of the options to apply_nan_suppression.

>>> import theano
>>> import theano.tensor as T
>>> import numpy as np
>>> from collections import OrderedDict
>>> from helpers import apply_nan_suppression

With `print_mode='shape'`, print only the shape of the update (not the entire
array contents).

>>> param = theano.shared(np.array([0., 0.]).astype(theano.config.floatX),
...                       name='param')
>>> inc = T.vector('inc')
>>> updates = OrderedDict([(param, param + inc)])
>>> safe_updates = apply_nan_suppression(updates, print_mode='shape')
>>> func = theano.function([inc], safe_updates[param],
...                        updates=safe_updates)
>>> func([1., 2.])
array([ 1.,  2.])
>>> func([2., float('nan')])
Warning: non-finite update suppressed for param: shape = (2,)
array([ 1.,  2.])

With `print_mode='none'`, don't print anything when NaNs are detected.

>>> safe_updates = apply_nan_suppression(updates, print_mode='none')
>>> func = theano.function([inc], safe_updates[param],
...                        updates=safe_updates)
>>> func([1., 2.])
array([ 2.,  4.])
>>> func([2., float('nan')])
array([ 2.,  4.])
'''

import doctest

if __name__ == '__main__':
    doctest.testmod()
