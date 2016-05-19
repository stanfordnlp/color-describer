import csv
import numpy as np
import sys
from collections import defaultdict


def print_report(infile):
    results = get_results(infile)
    print('Accuracy by example:')
    for system, correct, total in compare(results, ['HITId', 'Input.ex', 'Input.system']):
        print('    %s: %s/%s (%s%%)' % (system, correct, total, correct * 100.0 / total))
    print('Accuracy by user:')
    for system, correct, total in compare(results, ['WorkerId']):
        print('    %s: %s/%s (%s%%)' % (system, correct, total, correct * 100.0 / total))
    print('Accuracy by system:')
    for system, correct, total in compare(results, ['Input.system']):
        print('    %s: %s/%s (%s%%)' % (system, correct, total, correct * 100.0 / total))
    print('Accuracy by target:')
    by_answer = compare(results, ['Input.target'])
    for system, correct, total in by_answer:
        print('    %s: %s/%s (%s%%)' % (system, correct, total, correct * 100.0 / total))
    print('Accuracy by Turker answer:')
    by_answer = compare(results, ['Answer.answer'])
    for system, correct, total in by_answer:
        print('    %s: %s/%s (%s%%)' % (system, correct, total, correct * 100.0 / total))
    print('Accuracy by agreement:')
    by_agreement = compare_agreement(results)
    for system, correct, total in by_agreement:
        print('    %s: %s/%s (%s%%)' % (system, correct, total, correct * 100.0 / total))
    print('Counts of differences:')
    by_diffs = count_differences(results)
    for system, correct, total in by_diffs:
        print('    %s: %s/%s (%s%%)' % (system, correct, total, correct * 100.0 / total))
    print("Fleiss's kappa: %f" % fleiss_kappa(by_answer, by_agreement))
    print('Number of items: %d' % len(results))


def get_results(infile):
    reader = csv.DictReader(infile)
    rows = list(reader)
    items = []
    for row in rows:
        for ex in range(10):
            item = {}
            for k, v in row.items():
                new_k = remove_example_num(k, ex)
                if new_k:
                    item[new_k] = v
            item['Input.ex'] = str(ex)
            items.append(item)
    return items


def remove_example_num(key, example_num):
    '''
    >>> remove_example_num('Input.ex5target', 5)
    'Input.target'
    >>> remove_example_num('Input.ex5target', 7)
    >>> remove_example_num('WorkerId', 7)
    'WorkerId'
    '''
    pos = -1
    while pos < len(key) and not key[pos + 3].isdigit():
        pos = key.find('.ex', pos + 1)
        if pos == -1:
            break
    if pos == -1:
        # No example number in key, include pair in every item
        return key

    end_pos = pos + 3
    while end_pos < len(key) and key[end_pos].isdigit():
        end_pos += 1

    if int(key[pos + 3:end_pos]) == example_num:
        return key[:pos + 1] + key[end_pos:]
    else:
        return None


def compare(results, groupby):
    cats = defaultdict(list)
    for item in results:
        key = tuple(item[f] for f in groupby)
        cats[key].append(item['Answer.answer'] == item['Input.target'])
    comparison = [(cat, correct.count(True), len(correct)) for cat, correct in cats.items()]
    percent_correct = lambda (cat, num_correct, total): (-num_correct * 100.0 / total, cat)
    return sorted(comparison, key=percent_correct)


def compare_agreement(results):
    answers = defaultdict(list)
    correct = {}
    for item in results:
        answers[item['HITId'], item['Input.ex']].append(item['Answer.answer'])
        correct[item['HITId'], item['Input.ex']] = item['Input.target']
    rows = [{'Agreement': agreement(answers[key]),
             'Input.target': correct[key],
             'Answer.answer': majority(answers[key])}
            for key in correct]
    return compare(rows, groupby=['Agreement'])


def count_differences(results):
    answers = defaultdict(lambda: defaultdict(list))
    correct = {}
    for item in results:
        answers[item['Input.cid']][item['Input.system']].append(item['Answer.answer'])
        correct[item['Input.cid']] = item['Input.target']
    rows = [{'Comparison': differences(answers[key], correct[key]),
             'Input.target': correct[key],
             'Answer.answer': majority([a for system in answers[key]
                                          for a in answers[key][system]])}
            for key in correct]
    return compare(rows, groupby=['Comparison'])


def agreement(answers):
    return [a == b
            for i, a in enumerate(answers)
            for j, b in enumerate(answers) if i < j].count(True)


def differences(systems, correct):
    result = []
    for sys, ans in systems.items():
        assert len(ans) == 1, ans
        result.append('%s:%s' % (sys, ('right' if ans[0] == correct else 'wrong')))
    return tuple(sorted(result))


def majority(answers):
    counts = {}
    for a in answers:
        counts[a] = answers.count(a)
    max_agreement = max(counts.values())
    return [a for a, agr in counts.items() if agr == max_agreement][0]


def fleiss_kappa(by_answer, by_agreement):
    answer_dist = np.array([total * 1.0 for _, _, total in by_answer])
    answer_dist /= np.sum(answer_dist)
    agreements = np.array([agr[0] / 10.0 for agr, _, total in by_agreement for _ in range(total)])
    mean_agreement = np.mean(agreements)
    chance_agreement = np.sum(answer_dist ** 2)
    print('Mean agreement: %f' % mean_agreement)
    print('Chance agreement: %f' % chance_agreement)
    return (mean_agreement - chance_agreement) / (1.0 - chance_agreement)

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as infile:
        print_report(infile)
