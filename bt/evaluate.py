import numpy as np
import bootstrap


def evaluate(learner, eval_data, metric, metric_name=None):
    if metric_name is None and hasattr(metric, '__name__'):
        metric_name = metric.__name__
    prefix = metric_name + '.' if metric_name else ''

    inst_outputs = metric(learner, eval_data)

    mean = np.mean(inst_outputs)
    std = np.std(inst_outputs)
    if std:
      ci_lower, ci_upper = bootstrap.ci(inst_outputs)
    else:
      ci_lower = ci_upper = mean

    return {
      prefix + 'mean': mean,
      prefix + 'std': std,
      prefix + 'ci_lower': ci_lower,
      prefix + 'ci_upper': ci_upper,
    }
