import numpy as np


def log_likelihood(eval_data, predictions, scores):
    return scores


def accuracy(eval_data, predictions, scores):
    return [int(inst.output == pred)
            for inst, pred in zip(eval_data, predictions)]


def squared_error(eval_data, predictions, scores):
    return [np.sum((np.array(pred) - np.array(inst.output)) ** 2)
            for inst, pred in zip(eval_data, predictions)]
