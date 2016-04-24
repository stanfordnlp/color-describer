import colorsys
import glob
import json
import numpy as np
import os
import warnings
from collections import namedtuple
from numbers import Number

from stanza.research import config, instance  # NOQA (for doctest)


parser = config.get_options_parser()
parser.add_argument('--compare_dir', type=str, default=None,
                    help='The directory containing other results providing a point for '
                         'comparison to the run_dir, if not None. These results will also '
                         'be included in the report.')

Output = namedtuple('Output', 'config,results,data,scores,predictions')

MAX_ALTS = 10


class NotPresent(object):
    def __repr__(self):
        return ''


def html_report(output, compare=None):
    '''
    >>> config_dict = {'run_dir': 'runs/test', 'listener': True}
    >>> results_dict = {'dev.perplexity.gmean': 14.0}
    >>> data = [instance.Instance([0.0, 100.0, 100.0], 'red').__dict__]
    >>> scores = [-2.639057329615259]
    >>> predictions = ['bright red']
    >>> print(html_report(Output(config_dict, results_dict, data, scores, predictions)))
    <html>
    <head>
    <link rel="stylesheet" href="http://web.stanford.edu/~wmonroe4/css/style.css" type="text/css">
    <title>runs/test - Output report</title>
    </head>
    <body>
        <h1>runs/test</h1>
        <p>Compared to: (no comparison set)</p>
        <h2>Configuration options</h2>
        <table>
            <tr><th>Option</th><th>Value</th></tr>
            <tr><td>listener</td><td>True</td></tr>
            <tr><td>run_dir</td><td>'runs/test'</td></tr>
        </table>
        <h2>Results</h2>
        <h3>dev</h3>
        <table>
            <tr><th>Metric</th><th>gmean</th></tr>
            <tr><td>perplexity</td><td align="right">14.000</td></tr>
        </table>
        <h2>Error analysis</h2>
        <h3>Worst</h3>
        <table>
            <tr><th>input</th><th>gold</th><th>prediction</th><th>prob</th></tr>
            <tr><td bgcolor="#ff0000">[0, 100, 100]</td><td bgcolor="#fff">'red'</td><td bgcolor="#fff">'bright red'</td><td>0.071</td></tr>
        </table>
        <h3>Best</h3>
        <table>
            <tr><th>input</th><th>gold</th><th>prediction</th><th>prob</th></tr>
            <tr><td bgcolor="#ff0000">[0, 100, 100]</td><td bgcolor="#fff">'red'</td><td bgcolor="#fff">'bright red'</td><td>0.071</td></tr>
        </table>
        <h3>Head</h3>
        <table>
            <tr><th>input</th><th>gold</th><th>prediction</th><th>prob</th></tr>
            <tr><td bgcolor="#ff0000">[0, 100, 100]</td><td bgcolor="#fff">'red'</td><td bgcolor="#fff">'bright red'</td><td>0.071</td></tr>
        </table>
    </body>
    </html>
    '''  # NOQA

    main_template = '''<html>
<head>
<link rel="stylesheet" href="http://web.stanford.edu/~wmonroe4/css/style.css" type="text/css">
<title>{run_dir} - Output report</title>
</head>
<body>
    <h1>{run_dir}</h1>
    <p>Compared to: {compare_dir}</p>
    <h2>Configuration options</h2>
    <table>
        <tr><th>Option</th><th>Value</th>{compare_header}</tr>
{config_opts}
    </table>
    <h2>Results</h2>
{results}
    <h2>Error analysis</h2>
{error_analysis}
</body>
</html>'''

    return main_template.format(
        run_dir=output.config['run_dir'],
        compare_dir=compare.config['run_dir'] if compare else '(no comparison set)',
        compare_header='<th>Comparison</th>' if compare else '',
        config_opts=format_config_dict(output.config, compare.config if compare else None),
        results=format_results(output.results, compare.results if compare else None),
        error_analysis=format_error_analysis(output, compare)
    )


def format_config_dict(this_config, compare_config):
    config_opt_template = '        <tr><td>{key}</td>{values}</tr>'
    config_value_template = '<td>{!r}</td>'
    all_keys = set(this_config.keys())
    dicts = [this_config]
    if compare_config:
        all_keys.update(compare_config.keys())
        dicts.append(compare_config)
    all_keys = sorted(all_keys)
    return '\n'.join(
        config_opt_template.format(
            key=k,
            values=''.join(
                config_value_template.format(safe_lookup(d, k))
                for d in dicts
            )
        )
        for k in all_keys
    )


def safe_lookup(d, key):
    if not d:
        return NotPresent
    if key not in d:
        return NotPresent
    return d[key]


def format_results(results, compare=None):
    # TODO: compare
    results_table_template = '''    <h3>{split}</h3>
    <table>
{header}
{rows}
    </table>'''
    header_template = '        <tr><th>Metric</th>{aggregates}</tr>'
    row_template = '        <tr><td>{metric}</td>{values}</tr>'

    splits = sorted(set(metric.split('.')[0] for metric in results.keys()))
    tables = []
    for split in splits:
        items = [i for i in results.items() if i[0].startswith(split + '.')]
        metrics = sorted(set(''.join(m.split('.')[1]) for m, v in items))
        aggregates = sorted(set(''.join(m.split('.')[2:]) for m, v in items))
        header = header_template.format(aggregates=''.join('<th>{}</th>'.format(a)
                                                           for a in aggregates))
        values_table = [
            [
                get_formatted_result(results, split, m, a)
                for a in aggregates
            ]
            for m in metrics
        ]
        rows = '\n'.join(
            row_template.format(metric=m, values=''.join('<td align="right">{}</td>'.format(v)
                                                         for v in row))
            for m, row in zip(metrics, values_table)
        )
        tables.append(results_table_template.format(split=split, header=header, rows=rows))
    return '\n'.join(tables)


def get_formatted_result(results, split, m, a):
    key = '.'.join((split, m, a) if a else (split, m))
    if key in results:
        return format_number(results[key])
    else:
        return ''


def format_number(value):
    if not isinstance(value, Number):
        return repr(value)
    elif isinstance(value, int):
        return '{:,d}'.format(value)
    elif value > 1e8 or abs(value) < 1e-3:
        return '{:.6e}'.format(value)
    else:
        return '{:,.3f}'.format(value)


def format_error_analysis(output, compare=None):
    examples_table_template = '''    <h3>{cond}</h3>
    <table>
        <tr><th>input</th>{alt_inputs_header}{alt_outputs_header}<th>gold</th><th>prediction</th><th>prob</th>{compare_header}</tr>
{examples}
    </table>'''

    example_template = '        <tr>{input}{alt_inputs}{alt_outputs}{output}' \
                       '{prediction}{pprob}{comparison}{cprob}</tr>'
    score_template = '<td>{}</td>'
    show_alt_inputs = max_len(output.data, 'alt_inputs')
    show_alt_outputs = max_len(output.data, 'alt_outputs')
    collated = []
    for i, (inst, score, pred) in enumerate(zip(output.data, output.scores, output.predictions)):
        example = {}
        example['input'] = format_value(inst['input'])
        example['alt_inputs'] = format_alts(inst['alt_inputs'], show_alt_inputs)
        example['output'] = format_value(inst['output'])
        example['alt_outputs'] = format_alts(inst['alt_outputs'], show_alt_outputs)
        example['prediction'] = format_value(pred)
        pprob = np.exp(score) if isinstance(score, Number) else score
        example['pprob'] = score_template.format(format_number(pprob))
        example['pprob_val'] = pprob if isinstance(pprob, Number) else 0
        if compare:
            if compare.data[i]['input'] == inst['input']:
                example['comparison'] = format_value(compare.predictions[i])
                cscore = compare.scores[i]
                cprob = np.exp(cscore) if isinstance(cscore, Number) else cscore
                example['cprob'] = score_template.format(format_number(cprob))
                example['cprob_val'] = cprob if isinstance(cprob, Number) else 0
            else:
                warnings.warn("Comparison input doesn't match this input: %s != %s" %
                              (compare.data[i]['input'], inst['input']))
                example['comparison'] = ''
                example['cprob'] = ''
        else:
            example['comparison'] = ''
            example['cprob'] = ''
        collated.append(example)

    score_order = sorted(collated, key=lambda e: e['pprob_val'])
    tables = [
        ('Worst', score_order[:100]),
        ('Best', reversed(score_order[-100:])),
        ('Head', collated[:100]),
    ]
    if compare:
        diff_order = sorted(collated, key=lambda e: e['pprob_val'] - e['cprob_val'])
        tables.extend([
            ('Biggest decline', diff_order[:100]),
            ('Biggest improvement', reversed(diff_order[-100:])),
        ])

    return '\n'.join(examples_table_template.format(
        cond=cond,
        alt_inputs_header=(('<th>alt inputs</th>' if show_alt_inputs else '') +
                           '<th></th>' * (show_alt_inputs - 1)),
        alt_outputs_header=(('<th>alt outputs</th>' if show_alt_outputs else '') +
                            '<th></th>' * (show_alt_outputs - 1)),
        compare_header='<th>comparison</th><th>prob</th>' if compare else '',
        examples='\n'.join(
            example_template.format(**inst) for inst in examples
        )
    ) for cond, examples in tables)


def format_value(value):
    if isinstance(value, list):
        color = web_color(value)
        value = [int(c) for c in value]
    else:
        color = '#fff'
    return '<td bgcolor="{color}">{value!r}</td>'.format(color=color, value=value)


def format_alts(alts, num_alts):
    if not alts:
        alts = []
    alts = alts[:num_alts]
    alts = alts + [NotPresent()] * (num_alts - len(alts))
    return ''.join(format_value(v) for v in alts)


def web_color(hsv):
    '''
    >>> web_color((0.0, 100.0, 100.0))
    '#ff0000'
    >>> web_color((120.0, 50.0, 50.0))
    '#408040'
    '''
    hue, sat, val = hsv
    hsv_0_1 = (hue / 360., sat / 100., val / 100.)
    rgb = colorsys.hsv_to_rgb(*hsv_0_1)
    rgb_int = tuple(min(int(c * 256.0), 255) for c in rgb)
    return '#%02x%02x%02x' % rgb_int


def max_len(insts, field):
    result = 0
    for inst in insts:
        result = max(result, len(inst[field]) if inst[field] else 0)
    return min(MAX_ALTS, result)


def generate_html_reports(run_dir=None, compare_dir=None):
    options = config.options(read=True)
    run_dir = run_dir or options.run_dir
    compare_dir = compare_dir or options.compare_dir

    for output, compare, out_path in get_all_outputs(run_dir, options.compare_dir):
        with open(out_path, 'w') as outfile:
            outfile.write(html_report(output, compare))


def get_all_outputs(run_dir, compare_dir):
    for filename in glob.glob(os.path.join(run_dir, 'data.*.jsons')):
        split = os.path.basename(filename).split('.')[-2]
        this_output = get_output(run_dir, split)
        if compare_dir:
            compare = get_output(compare_dir, split)
        else:
            compare = None

        out_path = os.path.join(run_dir, 'report.%s.html' % split)
        yield this_output, compare, out_path


def get_output(run_dir, split):
    config_dict = load_dict(os.path.join(run_dir, 'config.json'))

    results = {}
    for filename in glob.glob(os.path.join(run_dir, 'results.*.json')):
        results.update(load_dict(filename))

    data = load_dataset(os.path.join(run_dir, 'data.%s.jsons' % split))
    scores = load_dataset(os.path.join(run_dir, 'scores.%s.jsons' % split))
    predictions = load_dataset(os.path.join(run_dir, 'predictions.%s.jsons' % split))
    return Output(config_dict, results, data, scores, predictions)


def load_dict(filename):
    try:
        with open(filename) as infile:
            return json.load(infile)
    except IOError, e:
        warnings.warn(str(e))
        return {'error.message.value': str(e)}


def load_dataset(filename, transform_func=(lambda x: x)):
    try:
        dataset = []
        with open(filename) as infile:
            for line in infile:
                js = json.loads(line.strip())
                dataset.append(transform_func(js))
        return dataset
    except IOError, e:
        warnings.warn(str(e))
        return [{'error': str(e)}]


if __name__ == '__main__':
    generate_html_reports()
