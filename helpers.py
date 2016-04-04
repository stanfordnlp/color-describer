import numpy as np  # NOQA: for doctest
import theano  # NOQA: for doctest
import theano.tensor as T
from collections import OrderedDict
from theano.ifelse import ifelse
from theano.printing import Print


def apply_nan_suppression(updates, print_mode='all'):
    """Returns a modified update dictionary replacing updates containing
    non-finite values with no-op updates

    If any NaN or infinity values are found in the new_expression (second)
    half of an update, the update is replaced with the do-nothing update
    (shared_variable, shared_variable).

    This can be used to patch over the most intransigent, slippery instances
    of NaNs creeping into training, if they appear rarely and one is reasonably
    sure that the problem is not fundamental to the model.

    Parameters
    ----------
    updates : OrderedDict
        A dictionary mapping parameters to update expressions

    print_mode : str
        If ``'all'``, print a debugging message containing the name of the
        shared variable and its suppressed update value whenever a non-finite
        value is detected. If ``'shape'``, print only the name of the variable
        and the shape of the update value. If ``'none'``, suppress NaNs
        silently without printing anything.

    Returns
    -------
    OrderedDict
        A copy of `updates` with expressions containing non-finite values
        replaced by the original value.

    Examples
    --------
    >>> param = theano.shared(np.array([0., 0.], dtype=np.float32),
    ...                       name='param')
    >>> inc = T.fvector('inc')
    >>> updates = OrderedDict([(param, param + inc)])
    >>> safe_updates = apply_nan_suppression(updates)
    >>> func = theano.function([inc], safe_updates[param],
    ...                        updates=safe_updates)
    >>> func([1., 2.])
    array([ 1.,  2.], dtype=float32)
    >>> func([2., float('nan')])
    Warning: non-finite update suppressed for param: __str__ = [  3.  nan]
    array([ 1.,  2.], dtype=float32)
    """
    new_updates = OrderedDict([])

    for shared_variable, new_expression in updates.iteritems():
        isnan = T.isnan(new_expression).any() | T.isinf(new_expression).any()

        warning_msg = 'Warning: non-finite update suppressed for %s'
        if print_mode == 'all':
            suppressed = T.zeros_like(
                Print((warning_msg + ':') % shared_variable.name)(new_expression)
            )
        elif print_mode == 'shape':
            suppressed = T.zeros_like(
                Print((warning_msg + ':') % shared_variable.name,
                      attrs=('shape',))(new_expression)
            )
        elif print_mode == 'none' or print_mode is None:
            suppressed = T.zeros_like(new_expression)
        else:
            raise ValueError("print_mode must be one of 'all', 'shape', or 'none'")

        # For some reason, the ifelse needs to be used in a calculation, or the
        # Print gets optimized away. So we can't do
        #   suppressed = (zeros_like(Print('warning')(new_expression)) +
        #                 shared_variable)
        #   ifelse(isnan, suppressed, new_expression)
        new_updates[shared_variable] = shared_variable + ifelse(isnan, suppressed,
                                                                new_expression - shared_variable)

    return new_updates
