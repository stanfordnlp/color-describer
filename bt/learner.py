class Learner(object):
    def __init__(self):
        self.__using_default_separate = False
        self.__using_default_combined = False

    def train(self, training_instances):
        '''
        Fit a model on training data.

        :param training_instances: The data to use to train the model.
            Instances should have at least the `input` and `output` fields
            populated.
        :type training_instances: list(instance.Instance)

        :returns: None
        '''
        raise NotImplementedError

    def predict(self, eval_instances):
        '''
        Return most likely predictions for each testing instance in
        `eval_instances`.

        :param eval_instances: The data to use to evaluate the model.
            Instances should have at least the `input` field populated.
            The `output` field need not be populated; subclasses should
            ignore it if it is present.
        :type eval_instances: list(instance.Instance)

        :returns: list(output_type)
        '''
        if self.__using_default_combined:
            raise NotImplementedError

        self.__using_default_separate = True
        return self.predict_and_score(eval_instances)[0]

    def score(self, eval_instances):
        '''
        Return scores (negative log likelihoods) assigned to each testing
        instance in `eval_instances`.

        :param eval_instances: The data to use to evaluate the model.
            Instances should have at least the `input` and `output` fields
            populated. `output` is needed to define which score is to
            be returned.
        :type eval_instances: list(instance.Instance)

        :returns: list(float)
        '''
        if self.__using_default_combined:
            raise NotImplementedError

        self.__using_default_separate = True
        return self.predict_and_score(eval_instances)[1]

    def predict_and_score(self, eval_instances):
        '''
        Return most likely outputs and scores for the particular set of
        outputs given in `eval_instances`, as a tuple. Return value should
        be equivalent to the default implementation of

            return (self.predict(eval_instances), self.score(eval_instances))

        but subclasses can override this to combine the two calls and reduce
        duplicated work. Either the two separate methods or this one (or all
        of them) should be overridden.

        :param eval_instances: The data to use to evaluate the model.
            Instances should have at least the `input` and `output` fields
            populated. `output` is needed to define which score is to
            be returned.
        :type eval_instances: list(bt.instance.Instance)

        :returns: tuple(list(output_type), list(float))
        '''
        if self.__using_default_separate:
            raise NotImplementedError

        self.__using_default_combined = True
        return (self.predict(eval_instances), self.score(eval_instances))
