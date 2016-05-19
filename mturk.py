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
    try:
        use_source = 'alt_inputs' in dir_output.data[0]['source']
    except (ValueError, KeyError):
        use_source = False
    src = lambda inst: (inst['source'] if use_source else inst)
    return [instance.Instance(pred, src(inst)['input'],
                              alt_outputs=src(inst)['alt_inputs'], source=tag)
            for inst, pred in zip(dir_output.data[:size], dir_output.predictions[:size])]


def output_csv():
    options = config.options(read=True)

    output = html_report.get_output(options.run_dir, options.split)
    insts = get_trial_data(output, options.test_size, options.run_dir)

    print(','.join('ex%d%s' % (ex, part)
                   for ex in range(BATCH_SIZE)
                   for part in ['cid', 'system', 'desc', 'target', 'c1', 'c2', 'c3']))

    for i, batch in enumerate(iterators.iter_batches(insts, BATCH_SIZE)):
        batch = list(batch)
        if len(batch) != BATCH_SIZE:
            continue
        print(','.join('"%d:%d","%s","%s","%s","%s","%s","%s"' %
                       ((i, j, inst.source, inst.input, inst.output) +
                        tuple(html_report.web_color(c) for c in inst.alt_outputs[:3]))
                       for j, inst in enumerate(batch)))


if __name__ == '__main__':
    output_csv()
