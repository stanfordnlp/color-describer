class Learner(object):
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
        raise NotImplementedError

    def score(self, eval_instances):
        '''
        Return scores (negative log likelihoods) assigned to each testing
        instance in `eval_instances`.

        :param eval_instances: The data to use to evaluate the model.
            Instances should have at least the `input` and `output` fields
            populated. `output` is needed to 
        :type eval_instances: list(instance.Instance)

        :returns: list(float)
        '''
        raise NotImplementedError
