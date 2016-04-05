from collections import defaultdict, Counter
import numpy as np

from stanza.unstable.learner import Learner
from stanza.unstable import progress, config
from lux import LuxLearner
from listener import ListenerLearner, AtomicListenerLearner
from speaker import SpeakerLearner, AtomicSpeakerLearner
from vectorizers import BucketsVectorizer
from rsa import RSALearner


def new(key):
    '''
    Construct a new learner with the class named by `key`. A list
    of available learners is in the dictionary `LEARNERS`.
    '''
    return LEARNERS[key]()


class Histogram(object):
    '''
    >>> from stanza.unstable.instance import Instance as I
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

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
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

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
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
    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        self.num_params = 0

    def predict_and_score(self, eval_instances):
        predict = [(128, 128, 128)] * len(eval_instances)
        score = [-3.0 * np.log(256.0)] * len(eval_instances)
        return predict, score


class LookupLearner(Learner):
    def __init__(self):
        options = config.options()
        self.counters = defaultdict(Counter)
        if options.listener:
            res = options.listener_color_resolution
            hsv = options.listener_hsv
        else:
            res = options.speaker_color_resolution
            hsv = options.speaker_hsv
        self.res = res
        self.hsv = hsv
        self.init_vectorizer()

    def init_vectorizer(self):
        if self.res and self.res[0]:
            if len(self.res) == 1:
                self.res = self.res * 3
            self.color_vec = BucketsVectorizer(self.res, hsv=self.hsv)
            self.vectorize = lambda c: self.color_vec.vectorize(c, hsv=True)
            self.unvectorize = lambda c: self.color_vec.unvectorize(c, hsv=True)
            self.score_adjustment = -np.log((256.0 ** 3) / self.color_vec.num_types)
        else:
            self.vectorize = lambda c: c
            self.unvectorize = lambda c: c
            self.score_adjustment = 0.0

    @property
    def num_params(self):
        return sum(len(c) for c in self.counters.values())

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        options = config.options()
        for inst in training_instances:
            inp, out = inst.input, inst.output
            if options.listener:
                out = self.vectorize(out)
            else:
                inp = self.vectorize(inp)
            self.counters[inp][out] += 1

    def predict_and_score(self, eval_instances, random='ignored', verbosity=0):
        options = config.options()
        if options.verbosity + verbosity >= 2:
            print('Testing')
        predictions = []
        scores = []
        for inst in eval_instances:
            inp, out = inst.input, inst.output
            if options.listener:
                out = self.vectorize(out)
            else:
                inp = self.vectorize(inp)

            counter = self.counters[inp]
            highest = counter.most_common(1)
            if highest:
                if options.listener:
                    prediction = self.unvectorize(highest[0][0])
                else:
                    prediction = highest[0][0]
            elif options.listener:
                prediction = (0, 0, 0)
            else:
                prediction = '<unk>'

            total = sum(counter.values())
            if total:
                if options.verbosity + verbosity >= 9:
                    print('%s -> %s: %s of %s [%s]' % (repr(inp), repr(out), counter[out],
                                                       total, inst.input))
                prob = counter[out] * 1.0 / total
            else:
                if options.verbosity + verbosity >= 9:
                    print('%s -> %s: no data [%s]' % (repr(inp), repr(out), inst.input))
                prob = 1.0 * (inst.output == prediction)
            score = np.log(prob)
            if options.listener:
                score += self.score_adjustment

            predictions.append(prediction)
            scores.append(score)

        return predictions, scores

    def __getstate__(self):
        return {
            'counters': {k: dict(v) for k, v in self.counters.iteritems()},
            'res': self.res,
            'hsv': self.hsv,
        }

    def __setstate__(self, state):
        self.res = state['res']
        self.hsv = state['hsv']
        self.init_vectorizer()
        self.counters = defaultdict(Counter, {k: Counter(v) for k, v in state['counters']})


LEARNERS = {
    'Histogram': HistogramLearner,
    'Lux': LuxLearner,
    'Listener': ListenerLearner,
    'AtomicListener': AtomicListenerLearner,
    'Speaker': SpeakerLearner,
    'AtomicSpeaker': AtomicSpeakerLearner,
    'RSA': RSALearner,
    'MostCommon': MostCommonSpeakerLearner,
    'Random': RandomListenerLearner,
    'Lookup': LookupLearner,
}

# Break cyclic dependency: ExhaustiveS1Learner needs list of learners to define
# exhaustive_base_learner command line option, LEARNERS needs ExhaustiveS1Learner
# to be defined to include it in the list.
import ref_game
LEARNERS.update({
    'ExhaustiveS1': ref_game.ExhaustiveS1Learner,
})
