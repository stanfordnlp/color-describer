from stanza.research import config, metrics, instance
from stanza.research.learner import Learner
import html_report


parser = config.get_options_parser()
parser.add_argument('--splits', type=str, default=['dev'],
                    help='Which data splits to output a results file for.')
parser.add_argument('--metrics', type=str, choices=metrics.METRICS.keys(), nargs='+',
                    default=['log_likelihood',
                             'log_likelihood_bits',
                             'perplexity',
                             'aic_averaged'],
                    help='Which metrics to output a results file for.')


def write_metrics():
    options = config.options(read=True)

    for split in options.splits:
        output = html_report.get_output(options.run_dir, split)
        for m in options.metrics:
            write_metric_for_split(output, options.run_dir, split, m)


def write_metric_for_split(output, run_dir, split, metric_name):
    filename = '%s.%s.jsons' % (metric_name, split)
    learner = Learner()
    learner.num_params = output.results['%s.num_params' % split]
    metric_func = metrics.METRICS[metric_name]
    if output.data[0].keys() == ['error']:
        data_insts = []
    else:
        data_insts = (instance.Instance(**d) for d in output.data)
    metric_scores = metric_func(data_insts, output.predictions, output.scores, learner)
    config.dump(metric_scores, filename, lines=True)


if __name__ == '__main__':
    write_metrics()
