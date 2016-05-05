from glob import glob
from sklearn.cross_validation import KFold

from stanza.research import config
from stanza.research.instance import Instance
from stanza.research.rng import get_rng

from tuna import TunaCorpus

parser = config.get_options_parser()
parser.add_argument('--tuna_section', default='*',
                    help='The name of the subset of the TUNA corpus to load, if `data_source` '
                         'is set to `tuna`; for example, "people". Asterisks can be '
                         'used to include multiple subsets.')
parser.add_argument('--tuna_cv_folds', type=int, default=5,
                    help='The number of cross-validation folds to use for the TUNA data.')
parser.add_argument('--tuna_cv_test_fold', type=int, default=4,
                    help='The index of the cross-validation fold to use as the test set '
                         'for the TUNA data. From 0 to `tuna_cv_folds` - 1, inclusive.')

rng = get_rng()

_trials_map = {}


def get_tuna_insts(files_glob, cv_folds):
    if (files_glob, cv_folds) not in _trials_map:
        filenames = glob(files_glob)
        if not filenames:
            raise IOError('Could not find any TUNA trials in "%s"' % files_glob)
        corpus = TunaCorpus(filenames)
        trials = list(corpus.iter_trials())
        splits = list(KFold(n=len(trials), n_folds=cv_folds, shuffle=True, random_state=rng))
        _trials_map[files_glob, cv_folds] = trials, splits
    return _trials_map[files_glob, cv_folds]


def trials_to_insts(trials, listener=False):
    insts = []
    for trial in trials:
        desc = tuple(d.string_description for d in trial.descriptions)
        desc_attrs = tuple(tuple(sorted(set([str(a) for a in d.attribute_set])))
                           for d in trial.descriptions)
        targets = tuple(i for i, e in enumerate(trial.entities) if e.is_target())
        alt_referents = tuple(tuple(str(a) for a in e.attributes) for e in trial.entities)
        if listener:
            insts.append(Instance(input=desc, annotated_input=desc_attrs,
                                  output=targets, alt_outputs=alt_referents,
                                  source=trial))
        else:
            insts.append(Instance(input=targets, alt_inputs=alt_referents,
                                  output=desc, annotated_output=desc_attrs,
                                  source=trial))
    return insts


def tuna_train_cv(listener=False):
    options = config.options()
    files_glob = 'tuna/corpus/%s/*.xml' % (options.tuna_section,)
    trials, splits = get_tuna_insts(files_glob, options.tuna_cv_folds)
    train_indices, test_indices = splits[options.tuna_cv_test_fold]
    return trials_to_insts([trials[i] for i in train_indices], listener=listener)


def tuna_test_cv(listener=False):
    options = config.options()
    files_glob = 'tuna/corpus/%s/*.xml' % (options.tuna_section,)
    trials, splits = get_tuna_insts(files_glob, options.tuna_cv_folds)
    train_indices, test_indices = splits[options.tuna_cv_test_fold]
    return trials_to_insts([trials[i] for i in test_indices], listener=listener)


def tuna_all(listener=False, corpus='tuna/corpus'):
    options = config.options()
    files_glob = '%s/%s/*.xml' % (corpus, options.tuna_section,)
    trials, _ = get_tuna_insts(files_glob, options.tuna_cv_folds)
    return trials_to_insts(trials, listener=listener)


def tuna08_train(listener=False):
    return tuna_all(listener=listener, corpus='tuna/train')


def tuna08_dev(listener=False):
    return tuna_all(listener=listener, corpus='tuna/dev')


def tuna08_test(listener=False):
    return tuna_all(listener=listener, corpus='tuna/test')
