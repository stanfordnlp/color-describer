import numpy as np

from rugstk.core.lux import LUX
from stanza.unstable import progress
from stanza.unstable.learner import Learner


class LuxLearner(Learner):
    def train(self, training_instances):
        self.lux = LUX()

    def predict(self, eval_instances, random='ignored', verbosity='ignored'):
        predictions = []
        progress.start_task('Predict example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)
            predictions.append(self.lux.predict(inst.input)[0])
        progress.end_task()
        return predictions

    def score(self, eval_instances, verbosity='ignored'):
        scores = []
        progress.start_task('Score example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)
            scores.append(np.log(self.lux.posterior_likelihood(inst.input, inst.output)))
        progress.end_task()
        return scores

    @property
    def num_params(self):
        total = 0
        for label in self.lux.all:
            params = self.lux.get_params(label)
            for row in params:
                assert isinstance(row[0], float), row[0]
                total += len(row)
        return total

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        self.train([])
