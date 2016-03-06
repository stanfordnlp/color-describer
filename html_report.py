import colorsys
import glob
import json
import os

from stanza.unstable import config, instance  # NOQA (for doctest)


def html_report(config_dict, results, data, scores, predictions):
    '''
    >>> config_dict = {'run_dir': 'runs/test', 'listener': True}
    >>> results_dict = {'dev.perplexity.gmean': 14.0}
    >>> data = [instance.Instance([0.0, 100.0, 100.0], 'red').__dict__]
    >>> scores = [-2.639057329615259]
    >>> predictions = ['bright red']
    >>> print(html_report(config_dict, results_dict, data, scores, predictions))
    <html>
    <head>
    <link rel="stylesheet" href="http://web.stanford.edu/~wmonroe4/css/style.css" type="text/css">
    <title>runs/test - Output report</title>
    </head>
    <body>
        <h1>runs/test</h1>
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
            <tr><td>perplexity</td><td align="right">14.00</td></tr>
        </table>
        <h2>Error analysis</h2>
        <h3>Worst</h3>
        <table>
            <tr><th>input</th><th>output</th><th>prediction</th><th>score</th></tr>
            <tr><td bgcolor="#ff0000">[0, 100, 100]</td><td bgcolor="#fff">'red'</td><td bgcolor="#fff">'bright red'</td><td>-2.639057329615259</td></tr>
        </table>
        <h3>Best</h3>
        <table>
            <tr><th>input</th><th>output</th><th>prediction</th><th>score</th></tr>
            <tr><td bgcolor="#ff0000">[0, 100, 100]</td><td bgcolor="#fff">'red'</td><td bgcolor="#fff">'bright red'</td><td>-2.639057329615259</td></tr>
        </table>
        <h3>Head</h3>
        <table>
            <tr><th>input</th><th>output</th><th>prediction</th><th>score</th></tr>
            <tr><td bgcolor="#ff0000">[0, 100, 100]</td><td bgcolor="#fff">'red'</td><td bgcolor="#fff">'bright red'</td><td>-2.639057329615259</td></tr>
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
    <h2>Configuration options</h2>
    <table>
        <tr><th>Option</th><th>Value</th></tr>
{config_opts}
    </table>
    <h2>Results</h2>
{results}
    <h2>Error analysis</h2>
{error_analysis}
</body>
</html>'''

    return main_template.format(
        run_dir=config_dict['run_dir'],
        config_opts=format_config_dict(config_dict),
        results=format_results(results),
        error_analysis=format_error_analysis(data, scores, predictions)
    )


def format_config_dict(config_dict):
    config_opt_template = '        <tr><td>{0}</td><td>{1!r}</td></tr>'
    return '\n'.join(config_opt_template.format(*i) for i in sorted(config_dict.items()))


def format_results(results):
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
        value = results[key]
        if isinstance(value, int):
            return '{:,d}'.format(value)
        elif value > 1e8:
            return '{:.5e}'.format(value)
        else:
            return '{:,.2f}'.format(value)
    else:
        return ''


def format_error_analysis(data, scores, predictions):
    examples_table_template = '''    <h3>{cond}</h3>
    <table>
        <tr><th>input</th><th>output</th><th>prediction</th><th>score</th></tr>
{examples}
    </table>'''

    example_template = ('        <tr><td bgcolor="{icolor}">{input!r}</td>'
                        '<td bgcolor="{ocolor}">{output!r}</td>'
                        '<td bgcolor="{pcolor}">{prediction!r}</td>'
                        '<td>{score!r}</td></tr>')
    collated = [dict(inst) for inst in data]
    for inst, score, pred in zip(collated, scores, predictions):
        inst['score'] = score
        inst['prediction'] = pred
        if isinstance(inst['input'], list):
            inst['icolor'] = web_color(inst['input'])
            inst['input'] = [int(c) for c in inst['input']]
        else:
            inst['icolor'] = '#fff'
        if isinstance(inst['output'], list):
            inst['ocolor'] = web_color(inst['output'])
            inst['output'] = [int(c) for c in inst['output']]
        else:
            inst['ocolor'] = '#fff'
        if isinstance(inst['prediction'], list):
            inst['pcolor'] = web_color(inst['prediction'])
            inst['prediction'] = [int(c) for c in inst['prediction']]
        else:
            inst['pcolor'] = '#fff'

    score_order = sorted(collated, key=lambda i: i['score'])
    tables = [
        ('Worst', score_order[:100]),
        ('Best', reversed(score_order[-100:])),
        ('Head', collated[:100]),
    ]

    return '\n'.join(examples_table_template.format(
        cond=cond,
        examples='\n'.join(
            example_template.format(**inst) for inst in examples
        )
    ) for cond, examples in tables)


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


def generate_html_reports(dirname=None):
    options = config.options(read=True)
    dirname = dirname or options.run_dir

    config_dict = load_dict(os.path.join(dirname, 'config.json'))
    results = {}
    for filename in glob.glob(os.path.join(dirname, 'results.*.json')):
        results.update(load_dict(filename))
    for filename in glob.glob(os.path.join(dirname, 'data.*.jsons')):
        split = os.path.basename(filename).split('.')[-2]
        data = load_dataset(filename)
        scores = load_dataset(os.path.join(dirname, 'scores.%s.jsons' % split))
        predictions = load_dataset(os.path.join(dirname, 'predictions.%s.jsons' % split))
        with open(os.path.join(dirname, 'report.%s.html' % split), 'w') as outfile:
            outfile.write(html_report(config_dict, results, data, scores, predictions))


def load_dict(filename):
    try:
        with open(filename) as infile:
            return json.load(infile)
    except IOError, e:
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
        return [{'error': str(e)}]


if __name__ == '__main__':
    generate_html_reports()
