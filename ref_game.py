import numpy as np
from scipy.misc import logsumexp

from stanza.unstable import config, instance, progress, iterators
from stanza.unstable.learner import Learner

import learners
import vectorizers


parser = config.get_options_parser()
parser.add_argument('--exhaustive_base_learner', default='Listener',
                    choices=learners.LEARNERS.keys(),
                    help='The name of the model to use as the L0 for exhaustive enumeration-based '
                         'speaker models.')


class ExhaustiveS1Learner(Learner):
    def __init__(self):
        options = config.options()
        self.listener = learners.new(options.exhaustive_base_learner)

    def train(self, training_instances, validation_instances=None, metrics=None):
        return self.listener.train(training_instances=training_instances,
                                   validation_instances=validation_instances, metrics=metrics)

    @property
    def num_params(self):
        return self.listener.num_params

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        options = config.options()
        predictions = []
        scores = []

        all_utts = self.listener.seq_vec.tokens
        sym_vec = vectorizers.SymbolVectorizer()
        sym_vec.add_all(all_utts)

        true_batch_size = options.listener_eval_batch_size / len(all_utts)
        batches = iterators.iter_batches(eval_instances, true_batch_size)
        num_batches = (len(eval_instances) - 1) // true_batch_size + 1

        if options.verbosity + verbosity >= 2:
            print('Testing')
        progress.start_task('Eval batch', num_batches)
        for batch_num, batch in enumerate(batches):
            progress.progress(batch_num)
            batch = list(batch)
            output_grid = [instance.Instance(utt, inst.input)
                           for inst in batch for utt in sym_vec.tokens]
            true_indices = sym_vec.vectorize_all([inst.input for inst in batch])
            if len(true_indices.shape) == 2:
                # Sequence vectorizer; we're only using single tokens for now.
                true_indices = true_indices[:, 0]
            scores = self.listener.score(output_grid, verbosity=verbosity)
            # Normalize across utterances. Note that the listener returns probability
            # densities over colors.
            log_probs = np.array(scores).reshape((-1, len(all_utts)))
            log_probs -= logsumexp(log_probs, axis=1)[:, np.newaxis]
            pred_indices = np.argmax(log_probs, axis=1)
            predictions.extend(sym_vec.unvectorize_all(pred_indices))
            scores.extend(log_probs[np.arange(len(batch)), true_indices].tolist())
        progress.end_task()

        return predictions, scores

    def dump(self, outfile):
        return self.listener.dump(outfile)

    def load(self, infile):
        return self.listener.load(infile)
