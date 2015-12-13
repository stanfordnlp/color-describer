from bt.learner import Learner


class LookupLearner(Learner):
    def train(self, training_instances):
        # TODO
        pass

    def predict(self, eval_instances):
        # TODO
        return ['black' for inst in eval_instances]

    def score(self, eval_instances):
        # TODO
        return [0.0 for inst in eval_instances]
