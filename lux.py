import multiprocessing
import numpy as np

from rugstk.core.lux import LUX
from stanza.unstable import progress, config
from stanza.unstable.learner import Learner


parser = config.get_options_parser()
parser.add_argument('--lux_threads', type=int, default=8,
                    help='Number of threads to use for Lux model evaluation')
parser.add_argument('--lux_batch_size', type=int, default=1000,
                    help='Number of examples per batch for Lux model evaluation')


lux_ = None


def get_lux():
    global lux_
    if lux_ is None:
        lux_ = LUX()
    return lux_


def lux_predict_and_score(inst):
    return (get_lux().predict(inst.input)[0],
            np.log(lux_.posterior_likelihood(inst.input, inst.output)))


class LuxLearner(Learner):
    def train(self, training_instances):
        pass

    def predict_and_score(self, eval_instances, random='ignored', verbosity='ignored'):
        options = config.options()
        predictions = []
        scores = []
        pool = multiprocessing.Pool(options.lux_threads)
        batch_size = options.lux_batch_size

        progress.start_task('Example', len(eval_instances))
        for start in range(0, len(eval_instances), batch_size):
            progress.progress(start)
            batch_output = pool.map(lux_predict_and_score,
                                    eval_instances[start:start + batch_size])
            batch_preds, batch_scores = zip(*batch_output)
            predictions.extend(batch_preds)
            scores.extend(batch_scores)
        progress.end_task()

        return predictions, scores

    @property
    def num_params(self):
        total = 0
        lux = get_lux()
        for label in lux.all:
            params = lux.get_params(label)
            for row in params:
                assert isinstance(row[0], float), row[0]
                total += len(row)
        return total

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        self.train([])
