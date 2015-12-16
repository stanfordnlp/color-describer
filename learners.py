from collections import defaultdict, Counter
import numpy as np
import json

from bt.learner import Learner
from bt import timing


class Histogram(object):
    '''
    >>> from bt.instance import Instance as I
    >>> data = [I((0.0, 100.0, 49.0), 'red'),
    ...         I((0.0, 100.0, 45.0), 'dark red'),
    ...         I((240.0, 100.0, 49.0), 'blue')]
    >>> h = Histogram(data, names=['red', 'dark red', 'blue'],
    ...               granularity=(90, 10, 10))
    >>> h.get_probs((1.0, 101.0, 48.0))
    [0.5, 0.5, 0.0]
    >>> h.get_probs((240.0, 100.0, 40.0))
    [0.0, 0.0, 1.0]
    '''
    def __init__(self, training_instances, names, granularity=(1, 1, 1)):
        self.names = names
        self.buckets = defaultdict(Counter)
        self.bucket_counts = defaultdict(int)
        self.granularity = granularity

        self.add_data(training_instances)

    def add_data(self, training_instances):
        timing.start_task('Example', len(training_instances))
        for i, inst in enumerate(training_instances):
            timing.progress(i)
            bucket = self.get_bucket(inst.input)
            self.buckets[bucket][inst.output] += 1
            self.bucket_counts[bucket] += 1
        timing.end_task()

    def get_bucket(self, color):
        '''
        >>> Histogram([], [], granularity=(3, 5, 7)).get_bucket((0, 1, 2))
        (0, 0, 0)
        >>> Histogram([], [], granularity=(3, 5, 7)).get_bucket((5.5, 27.3, 7.0))
        (3, 25, 7)
        '''
        return tuple(
            g * int(d // g)
            for d, g in zip(color, self.granularity)
        )

    def get_probs(self, color):
        bucket = self.get_bucket(color)
        counter = self.buckets[bucket]
        bucket_size = self.bucket_counts[bucket]
        return [
            (counter[name] * 1.0 / bucket_size
             if bucket_size != 0
             else 1.0 / len(self.names))
            for name in self.names
        ]


class HistogramLearner(Learner):
    '''
    The histogram model (HM) baseline from section 5.1 of McMahan and Stone
    (2015).
    '''

    WEIGHTS = [0.322, 0.643, 0.035]
    GRANULARITY = [(90, 10, 10), (45, 5, 5), (1, 1, 1)]

    def __init__(self):
        self.hists = []
        self.names = []
        self.name_to_index = defaultdict(lambda: -1)

    def train(self, training_instances):
        self.names = sorted(set(inst.output for inst in training_instances)) + ['<unk>']
        self.name_to_index = defaultdict(lambda: -1,
                                         {n: i for i, n in enumerate(self.names)})
        self.hists = []
        timing.start_task('Histogram', len(self.GRANULARITY))
        for i, g in enumerate(self.GRANULARITY):
            timing.progress(i)
            self.hists.append(Histogram(training_instances, self.names, granularity=g))
        timing.end_task()

    def hist_probs(self, color):
        assert self.hists, \
            'No histograms constructed yet; calling predict/score before train?'

        probs = [np.array(h.get_probs(color)) for h in self.hists]
        return sum(w * p for w, p in zip(self.WEIGHTS, probs))

    def predict(self, eval_instances):
        return self.predict_and_score(eval_instances)[0]

    def score(self, eval_instances):
        return self.predict_and_score(eval_instances)[1]

    def predict_and_score(self, eval_instances):
        predict = []
        score = []
        timing.start_task('Example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            timing.progress(i)
            hist_probs = self.hist_probs(inst.input)
            name = self.names[hist_probs.argmax()]
            prob = hist_probs[self.name_to_index[inst.output]]
            predict.append(name)
            score.append(-np.log(prob))
        timing.end_task()
        return predict, score
