def unsafe_jacobian(expression, wrt, consider_constant=None,
                    disconnected_inputs='raise'):
    """
    :type expression: Vector (1-dimensional) Variable
    :type wrt: Variable or list of Variables
    :param consider_constant: a list of expressions not to backpropagate
        through
    :type disconnected_inputs: string
    :param disconnected_inputs: Defines the behaviour if some of the variables
        in ``wrt`` are not part of the computational graph computing ``cost``
        (or if all links are non-differentiable). The possible values are:
        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.
    :return: either a instance of Variable or list/tuple of Variables
            (depending upon `wrt`) repesenting the jacobian of `expression`
            with respect to (elements of) `wrt`. If an element of `wrt` is not
            differentiable with respect to the output, then a zero
            variable is returned. The return value is of same type
            as `wrt`: a list/tuple or TensorVariable in all cases.
    """
    import theano
    from theano.tensor import arange
    from theano.gof import Variable
    from theano.gradient import format_as, grad

    # Check inputs have the right format
    assert isinstance(expression, Variable), \
        "tensor.jacobian expects a Variable as `expression`"
    assert expression.ndim < 2, \
        ("tensor.jacobian expects a 1 dimensional variable as "
         "`expression`. If not use flatten to make it a vector")

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    if expression.ndim == 0:
        # expression is just a scalar, use grad
        return format_as(using_list, using_tuple,
                         grad(expression,
                              wrt,
                              consider_constant=consider_constant,
                              disconnected_inputs=disconnected_inputs))

    def inner_function(*args):
        idx = args[0]
        expr = args[1]
        rvals = []
        for inp in args[2:]:
            rval = grad(expr[idx],
                        inp,
                        consider_constant=consider_constant,
                        disconnected_inputs=disconnected_inputs)
            rvals.append(rval)
        return rvals
    # Computing the gradients does not affect the random seeds on any random
    # generator used n expression (because during computing gradients we are
    # just backtracking over old values. (rp Jan 2012 - if anyone has a
    # counter example please show me)
    jacobs, updates = theano.scan(inner_function,
                                  sequences=arange(expression.shape[0]),
                                  non_sequences=[expression] + wrt)
    # NOTE: assertion removed here! may cause weird things to happen if the
    #       computational graph contains RandomStreams
    return format_as(using_list, using_tuple, jacobs)
