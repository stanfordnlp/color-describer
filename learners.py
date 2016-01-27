from collections import defaultdict, Counter
import numpy as np

from bt.learner import Learner
from bt import progress
from listener import ListenerLearner
from speaker import SpeakerLearner, AtomicSpeakerLearner
from rsa import RSALearner


def new(key):
    '''
    Construct a new learner with the class named by `key`. A list
    of available learners is in the dictionary `LEARNERS`.
    '''
    return LEARNERS[key]()


class Histogram(object):
    '''
    >>> from bt.instance import Instance as I
    >>> data = [I((0.0, 100.0, 49.0), 'red'),
    ...         I((0.0, 100.0, 45.0), 'dark red'),
    ...         I((240.0, 100.0, 49.0), 'blue')]
    >>> h = Histogram(data, names=['red', 'dark red', 'blue'],
    ...               granularity=(4, 10, 10))
    >>> h.get_probs((1.0, 91.0, 48.0))
    [0.5, 0.5, 0.0]
    >>> h.get_probs((240.0, 100.0, 40.0))
    [0.0, 0.0, 1.0]
    '''
    def __init__(self, training_instances, names,
                 granularity=(1, 1, 1), use_progress=False):
        self.names = names
        self.buckets = defaultdict(Counter)
        self.bucket_counts = defaultdict(int)
        self.granularity = granularity
        self.bucket_sizes = (360 // granularity[0],
                             100 // granularity[1],
                             100 // granularity[2])
        self.use_progress = use_progress

        self.add_data(training_instances)

    def add_data(self, training_instances):
        if self.use_progress:
            progress.start_task('Example', len(training_instances))

        for i, inst in enumerate(training_instances):
            if self.use_progress:
                progress.progress(i)

            bucket = self.get_bucket(inst.input)
            self.buckets[bucket][inst.output] += 1
            self.bucket_counts[bucket] += 1

        if self.use_progress:
            progress.end_task()

    def get_bucket(self, color):
        '''
        >>> Histogram([], [], granularity=(3, 5, 10)).get_bucket((0, 1, 2))
        (0, 0, 0)
        >>> Histogram([], [], granularity=(3, 5, 10)).get_bucket((172.0, 30.0, 75.0))
        (120, 20, 70)
        >>> Histogram([], [], granularity=(3, 5, 10)).get_bucket((360.0, 100.0, 100.0))
        (240, 80, 90)
        '''
        return tuple(
            s * min(int(d // s), g - 1)
            for d, s, g in zip(color, self.bucket_sizes, self.granularity)
        )

    def get_probs(self, color):
        bucket = self.get_bucket(color)
        counter = self.buckets[bucket]
        bucket_size = self.bucket_counts[bucket]
        probs = []
        for name in self.names:
            prob = ((counter[name] * 1.0 / bucket_size)
                    if bucket_size != 0
                    else (1.0 / len(self.names)))
            probs.append(prob)
        return probs

    @property
    def num_params(self):
        return sum(len(counter) for _name, counter in self.buckets.items())

    def __getstate__(self):
        # `defaultdict`s aren't pickleable. Turn them into regular dicts for pickling.
        state = dict(self.__dict__)
        for name in ('buckets', 'bucket_counts'):
            state[name] = dict(state[name])
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.buckets = defaultdict(Counter, self.buckets)
        self.bucket_counts = defaultdict(int, self.bucket_counts)


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
        progress.start_task('Histogram', len(self.GRANULARITY))
        for i, g in enumerate(self.GRANULARITY):
            progress.progress(i)
            self.hists.append(Histogram(training_instances, self.names,
                                        granularity=g, use_progress=True))
        progress.end_task()

        self.num_params = sum(h.num_params for h in self.hists)

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
        predictions = []
        scores = []
        progress.start_task('Example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)
            hist_probs = self.hist_probs(inst.input)
            name = self.names[hist_probs.argmax()]
            prob = hist_probs[self.name_to_index[inst.output]]
            predictions.append(name)
            scores.append(np.log(prob))
        progress.end_task()
        return predictions, scores

    def __getstate__(self):
        state = dict(self.__dict__)
        state['name_to_index'] = dict(state['name_to_index'])
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.name_to_index = defaultdict(lambda: -1, self.name_to_index)


class MostCommonSpeakerLearner(Learner):
    def __init__(self):
        self.seen = Counter()
        self.num_examples = 0

    def train(self, training_instances):
        progress.start_task('Example', len(training_instances))
        for i, inst in enumerate(training_instances):
            progress.progress(i)
            self.seen.update([inst.output])
        progress.end_task()
        self.num_examples += len(training_instances)

    @property
    def num_params(self):
        return len(self.seen)

    def predict_and_score(self, eval_instances):
        most_common = self.seen.most_common(1)[0][0]
        predict = [most_common] * len(eval_instances)
        score = []
        progress.start_task('Example', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            progress.progress(i)
            score.append(np.log(self._get_smoothed_prob(inst.output)))
        progress.end_task()
        return predict, score

    def _get_smoothed_prob(self, output):
        if output in self.seen and self.seen[output] > 1:
            return (self.seen[output] - 1.0) / self.num_examples
        else:
            return 1.0 * len(self.seen) / self.num_examples


class RandomListenerLearner(Learner):
    def train(self, training_instances):
        self.num_params = 0

    def predict_and_score(self, eval_instances):
        predict = [(128, 128, 128)] * len(eval_instances)
        score = [-3.0 * np.log(256.0)] * len(eval_instances)
        return predict, score


LEARNERS = {
    'Histogram': HistogramLearner,
    'Listener': ListenerLearner,
    'Speaker': SpeakerLearner,
    'AtomicSpeaker': AtomicSpeakerLearner,
    'RSA': RSALearner,
    'MostCommon': MostCommonSpeakerLearner,
    'Random': RandomListenerLearner,
}
