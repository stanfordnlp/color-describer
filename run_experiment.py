from stanza.unstable import pick_gpu, config

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


from stanza.unstable import evaluate, metrics, output, progress
import datetime
import learners
import color_instances

parser.add_argument('--learner', default='Histogram', choices=learners.LEARNERS.keys(),
                    help='The name of the model to use in the experiment.')
parser.add_argument('--load', metavar='MODEL_FILE', default=None,
                    help='If provided, skip training and instead load a pretrained model '
                         'from the specified path. If None or an empty string, train a '
                         'new model.')
parser.add_argument('--train_size', type=int, default=None,
                    help='The number of examples to use in training. '
                         'If None, use the whole training set.')
parser.add_argument('--test_size', type=int, default=None,
                    help='The number of examples to use in testing. '
                         'If None, use the whole dev/test set.')
parser.add_argument('--data_source', default='dev', choices=color_instances.SOURCES.keys(),
                    help='The type of data to use.')
parser.add_argument('--output_train_data', action='store_true',
                    help='If True, write out the training dataset (after cutting down to '
                         '`train_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--output_test_data', action='store_true',
                    help='If True, write out the evaluation dataset (after cutting down to '
                         '`test_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--listener', action='store_true',
                    help='If True, evaluate on listener accuracy (description -> color). '
                         'Otherwise evaluate on speaker accuracy (color -> description).')
parser.add_argument('--progress_tick', type=int, default=300,
                    help='The number of seconds between logging progress updates.')


def main():
    options = config.options()

    progress.set_resolution(datetime.timedelta(seconds=options.progress_tick))

    train_data = color_instances.SOURCES[options.data_source].train_data(
        listener=options.listener
    )[:options.train_size]
    test_data = color_instances.SOURCES[options.data_source].test_data(
        options.listener
    )[:options.test_size]

    learner = learners.new(options.learner)

    m = [metrics.log_likelihood,
         metrics.log_likelihood_bits,
         metrics.perplexity,
         metrics.aic]
    if options.listener:
        m.append(metrics.squared_error)
    else:
        m.append(metrics.accuracy)

    if options.load:
        with open(options.load, 'rb') as infile:
            learner.load(infile)
    else:
        learner.train(train_data)
        with open(config.get_file_path('model.p'), 'wb') as outfile:
            learner.dump(outfile)

        train_results = evaluate.evaluate(learner, train_data, metrics=m, split_id='train',
                                          write_data=options.output_train_data)
        output.output_results(train_results, 'train')

    test_results = evaluate.evaluate(learner, test_data, metrics=m, split_id='dev',
                                     write_data=options.output_test_data)
    output.output_results(test_results, 'dev')


if __name__ == '__main__':
    main()
