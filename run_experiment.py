from bt import pick_gpu, config

parser = config.get_options_parser()
parser.add_argument('--device', default=None,
                    help='The device to use in Theano ("cpu" or "gpu[0-n]"). If None, '
                         'pick a free-ish device automatically.')
options, extras = parser.parse_known_args()
if '-h' in extras or '--help' in extras:
    # If user is just asking for the options, don't scare them
    # by saying we're picking a GPU...
    pick_gpu.bind_theano('cpu')
else:
    pick_gpu.bind_theano(options.device)


from bt import evaluate, metrics, output, progress
import datetime
import learners
import color_instances

parser.add_argument('--learner', default='Histogram', choices=learners.LEARNERS.keys(),
                    help='The name of the model to use in the experiment.')
parser.add_argument('--train_size', type=int, default=None,
                    help='The number of examples to use in training. '
                         'If None, use the whole training set.')
parser.add_argument('--test_size', type=int, default=None,
                    help='The number of examples to use in testing. '
                         'If None, use the whole dev/test set.')
parser.add_argument('--listener', action='store_true',
                    help='If True, evaluate on listener accuracy (description -> color). '
                         'Otherwise evaluate on speaker accuracy (color -> description).')
parser.add_argument('--progress_tick', type=int, default=300,
                    help='The number of seconds between logging progress updates.')


def main():
    options = config.options()
    learner = learners.new(options.learner)

    progress.set_resolution(datetime.timedelta(seconds=options.progress_tick))

    train_data = color_instances.get_training_instances(
        listener=options.listener
    )[:options.train_size]

    learner.train(train_data)

    m = ([metrics.log_likelihood, metrics.squared_error]
         if options.listener else
         [metrics.log_likelihood, metrics.accuracy])
    train_results = evaluate.evaluate(learner, train_data, metrics=m, split_id='train')
    output.output_results(train_results, 'train')

    dev_data = color_instances.get_dev_instances(options.listener)[:options.test_size]
    dev_results = evaluate.evaluate(learner, dev_data, metrics=m, split_id='dev')
    output.output_results(dev_results, 'dev')


if __name__ == '__main__':
    main()
