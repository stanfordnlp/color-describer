from stanza.research import config, instance, iterators
import html_report


parser = config.get_options_parser()
parser.add_argument('--split', type=str, default='dev',
                    help='The data split to draw the human evaluation data from.')
parser.add_argument('--test_size', type=int, default=None,
                    help='The number of examples to use in human evaluation. '
                         'If None, use the whole dev/test set.')

BATCH_SIZE = 10


def get_trial_data(dir_output, size, tag):
    return [instance.Instance(pred, inst['input'], alt_outputs=inst['alt_inputs'], source=tag)
            for inst, pred in zip(dir_output.data[:size], dir_output.predictions[:size])]


def output_csv():
    options = config.options(read=True)

    output = html_report.get_output(options.run_dir, options.split)
    insts = get_trial_data(output, options.test_size, options.run_dir)

    print(','.join('ex%d%s' % (ex, part)
                   for ex in range(BATCH_SIZE)
                   for part in ['desc', 'c1', 'c2', 'c3']))

    for batch in iterators.iter_batches(insts, BATCH_SIZE):
        batch = list(batch)
        if len(batch) != BATCH_SIZE:
            continue
        print(','.join('"%s","%s","%s","%s"' %
                       ((inst.input,) +
                        tuple(html_report.web_color(c) for c in inst.alt_outputs[:3]))
                       for inst in batch))


if __name__ == '__main__':
    output_csv()
