Learning to Generate Compositional Color Descriptions
=====================================================

Code and supplementary material

Note that this repo is split off from a larger project still in development:
https://github.com/futurulus/coop-nets

Outputs tables
--------------

A full table of samples from the model that provided the examples in Table 1
and Table 3 is included at: ::

    outputs/color_samples.html

Dependencies
------------

You'll first need Python 2.7. Creating and activating a new virtualenv or
Anaconda environment is recommended. Then run this script to download data and
Python package dependencies: ::

    ./dependencies

The dependencies script is reasonably simple, so if this fails, it should be
possible to look at the script and manually perform the actions it specifies.

This code is written to be run on a Linux system; we've also tested it on Mac
OS X (but see "Troubleshooting": missing g++ will cause the program to run
impossibly slowly). The code is unlikely to run on Windows, but you're welcome
to try.

Usage
-----

The easiest way to see the model in action is to use the ``colordesc``
module: ::

    >>> import colordesc
    >>> describer = colordesc.ColorDescriber()
    >>> describer.describe((255, 0, 0))
    'red'

This loads a pickled Theano model, which may not be very robust to different
system configurations or very future-proof; if you run into problems with this,
read on.

Repickling from model params
----------------------------

The pickle files in the ``models`` directory, with the exception of
``lstm_fourier_quick.p``, contain only the parameters of the model, which
makes them more portable, but they require a bit more work: ::

    $ mkdir -p runs/lstm_fourier
    $ cp models/lstm_fourier.p runs/lstm_fourier/model.p
    $ python quickpickle.py --config models/lstm_fourier.config.json --run_dir runs/lstm_fourier
    $ python
    >>> import colordesc, cPickle as pickle
    >>> with open('runs/lstm_fourier/quickpickle.p', 'rb') as picklefile:
    ...     describer = colordesc.ColorDescriber(picklefile)
    >>> describer.describe((255, 0, 0))
    'red'

Running experiments
-------------------

To re-run the experiments from the paper (Table 2) with pre-trained models, use
the following command, where ``lstm_fourier`` (our best model) can be replaced
with any of the eight experiment configurations in the outputs/ directory: ::

    python run_experiment.py --config models/lstm_fourier.config.json \
                             --load models/lstm_fourier.p \
                             --progress_tick 10

This should take about 15 minutes on a typical new-ish machine. Look for these
metrics in the outputs to compare with Table 2: ::

    dev.perplexity.gmean
    dev.aic.sum
    dev.accuracy.mean

The results of the experiment, including predictions and log-likelihood scores,
will be logged to the directory ::

    runs/lstm_fourier

To retrain a model from scratch, supply only the config file: ::

    python run_experiment.py --config models/lstm_fourier.config.json

Note that this may require several days to train on CPU. A properly-configured
GPU usually takes a few hours and can be used by passing ``--device gpu0``. See

    http://deeplearning.net/software/theano/tutorial/using_gpu.html

for necessary configuration.

Troubleshooting
---------------

* Error messages of the form

    ``error: argument --...: invalid int value: '<pyhocon.config_tree.NoneValue
    object at ...>'``

  should be solved by making sure you're using pyhocon version 0.3.18; if this
  doesn't work, supplying a number for the argument should fix it. We've seen
  this with the arguments ``--train_size`` or ``--test_size``; to fix these,
  add ::

    --train_size 10000000 --test_size 10000000

* A warning message of the form

    ``WARNING (theano.configdefaults): g++ not detected ! Theano will be unable
    to execute optimized C-implementations (for both CPU and GPU) and will
    default to Python implementations. Performance will be severely degraded.
    To remove this warning, set Theano flags cxx to an empty string.``

  should be heeded. Otherwise even just running prediction will take a very
  long time (days). Check whether you can run ``g++`` from a terminal, or try
  changing the Theano cxx flag (in ~/.theanorc) to point to an alternative C++
  compiler on the system.

* If retrying a run after a previous error, you'll need to add the option
  ``--overwrite`` (or specify a different output directory with ``--run_dir
  DIR``).  The program will remind you of this if you forget.

* Very large dev perplexity (``dev.perplexity.gmean`` > 50) could indicate
  incompatible changes in the version of Lasagne or Theano (we've seen this
  with Lasagne 0.1). We've reproduced our main results using the development
  versions of Theano and Lasagne as of June 2, 2016:

    * https://github.com/Theano/Theano/tree/0693ce052725a15b502068a1490b0637216feb00
    * https://github.com/Lasagne/Lasagne/tree/8fe645d28b66f991d547e9b6a314251b8e84446a
