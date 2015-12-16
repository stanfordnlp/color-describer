def log_likelihood(eval_data, predictions, scores):
    return scores

def accuracy(eval_data, predictions, scores):
    return [int(inst.output == pred)
            for inst, pred in zip(eval_data, predictions)]
