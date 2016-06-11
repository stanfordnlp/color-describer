try:
    import wx
    from wx import Dialog
    import wx.lib.colourselect as csel
except ImportError, e:
    import warnings
    warnings.warn('Could not import wx; human_eval cannot be used: %s' % e)
from itertools import izip

from stanza.research import config, evaluate, instance, learner, metrics, output, rng
import html_report
from colorutils import rgb_to_hsv


parser = config.get_options_parser()
parser.add_argument('--split', type=str, default='dev',
                    help='The data split to draw the human evaluation data from.')
parser.add_argument('--test_size', type=int, default=None,
                    help='The number of examples to use in human evaluation. '
                         'If None, use the whole dev/test set.')

BATCH_SIZE = 10

random = rng.get_rng()


class HumanListener(learner.Learner):
    def __init__(self):
        self.num_params = 0
        self.memory = {}
        with open('human_listener.txt', 'r') as infile:
            for line in infile:
                if '\t' in line:
                    desc, color_str = line.strip().split('\t')
                    assert color_str.startswith('(') and color_str.endswith(')'), color_str
                    color = tuple(float(d) for d in color_str[1:-1].split(', '))
                    self.memory[desc] = color

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        all_utts = list(set(inst.input for inst in training_instances
                            if inst.input not in self.memory))
        random.shuffle(all_utts)
        for start in range(0, len(all_utts), BATCH_SIZE):
            batch = all_utts[start:start + BATCH_SIZE]
            for utt, color in izip(batch, self.request_batch(batch, start, len(all_utts))):
                line = '%s\t%s' % (utt, color)
                with open('human_listener.txt', 'a') as outfile:
                    outfile.write(line + '\n')
                print(line)
                self.memory[utt] = color

    def request_batch(self, descriptions, start, total):
        dlg = ColorPickerDialog(descriptions, start, total, parent=None)
        dlg.ShowModal()
        result = dlg.get_colors()
        dlg.Destroy()
        return result

    def predict_and_score(self, eval_instances):
        predictions = [self.memory[inst.input] for inst in eval_instances]
        scores = [float('inf')] * len(eval_instances)
        return predictions, scores


class ColorPickerDialog(Dialog):
    def __init__(self, descriptions, start, total, *args, **kwargs):
        super(ColorPickerDialog, self).__init__(*args, **kwargs)
        sizer = wx.FlexGridSizer(cols=3)
        text = '%s-%s of %s' % (start + 1, start + len(descriptions), total)
        sizer.AddMany([
            ((5, 0),),
            (wx.StaticText(self, -1, text), 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL),
            ((0, 0),),
        ])
        self.buttons = []
        for desc in descriptions:
            button = csel.ColourSelect(self, -1, size=(60, 40))
            self.buttons.append(button)
            sizer.AddMany([
                ((5, 0),),
                (wx.StaticText(self, -1, desc), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL),
                (button, 0, wx.ALL, 3),
            ])
        self.SetSizerAndFit(sizer)

    def get_colors(self):
        return [rgb_to_hsv(button.GetColour()) for button in self.buttons]


def get_trial_data(dir_output, size, tag):
    return [instance.Instance(pred, inst['input'], source=tag)
            for inst, pred in zip(dir_output.data[:size], dir_output.predictions[:size])]


def main():
    options = config.options(read=True)

    app = wx.App()  # NOQA: wx needs an App even if we're only showing a few modal dialogs

    this_output = html_report.get_output(options.run_dir, options.split)
    this_insts = get_trial_data(this_output, options.test_size, options.run_dir)

    if options.compare_dir:
        compare_output = html_report.get_output(options.compare_dir, options.split)
        compare_insts = get_trial_data(compare_output, options.test_size, options.run_dir)
    else:
        compare_insts = []

    all_insts = this_insts + compare_insts
    random.shuffle(all_insts)

    human = HumanListener()
    human.train(all_insts)

    m = [metrics.squared_error]

    test_results = evaluate.evaluate(human, this_insts, split_id='human_eval', metrics=m)
    output.output_results(test_results, options.run_dir)
    if compare_insts:
        test_results = evaluate.evaluate(human, compare_insts,
                                         split_id='human_eval_compare', metrics=m)
        output.output_results(test_results, options.compare_dir)


if __name__ == '__main__':
    main()
