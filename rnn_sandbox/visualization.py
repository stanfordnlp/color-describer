import numpy as np
from matplotlib import pyplot as plot


def npenumerate(arr):
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        yield (it.multi_index, it[0])
        it.iternext()


def plot_matrix(mat, axes=None, show=True,
                xlabels=None, ylabels=None, prefix=()):
    if axes is None:
        axes = plot.axes()
    if len(mat.shape) < 2:
        plot_matrix(mat[np.newaxis, :], xlabels, ylabels, prefix)
    elif len(mat.shape) == 2:
        if prefix:
            print (prefix)
        num_rows, num_cols = mat.shape
        extent = [-0.5, num_cols - 0.5, num_rows - 0.5, -0.5]
        im = axes.imshow(mat, extent=extent, origin='upper',
                         interpolation='none', cmap=plot.get_cmap('gnuplot'))
        if xlabels:
            axes.set_yticks(range(len(ylabels)))
            axes.set_yticklabels(ylabels)
        if ylabels:
            axes.set_xticks(range(len(xlabels)))
            axes.set_xticklabels(xlabels)
        for label in im.axes.xaxis.get_ticklabels():
            label.set_rotation(90)
        if show:
            axes.figure.show()
    else:
        for j in xrange(mat.shape[0]):
            plot_matrix(mat[j], xlabels, ylabels, prefix + (j,))


def print_matrix(mat):
    if isinstance(mat, list):
        print('list[%s]' % type(mat[0]))
        if mat and hasattr(mat[0], 'toarray'):
            mat = np.array([m.toarray() for m in mat])
        else:
            mat = np.array(mat)
    elif hasattr(mat, 'toarray'):
        print(type(mat))
        mat = mat.toarray()
    else:
        print(type(mat))
    print(mat.shape)
    plot_matrix(mat)
    for i, v in npenumerate(mat):
        print '%s %s' % (i, v)
