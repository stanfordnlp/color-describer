#!/usr/bin/env python

import re
from glob import iglob
from itertools import groupby
import xml.etree.ElementTree as ET
from tokenizers import basic_unigram_tokenizer

import subprocess


class TunaCorpus:
    def __init__(self, filenames):
        self.filenames = list(filenames)

    def iter_trials(self):
        for group in group_references(self.filenames):
            yield Trial(group)


def group_references(filenames):
    '''
    >>> list(group_references(['a-0.xml', 'a-1.xml', 'b-0.xml', 'b-1.xml']))
    [['a-0.xml', 'a-1.xml'], ['b-0.xml', 'b-1.xml']]
    '''
    filenames = sorted(filenames)
    for _, group in groupby(filenames, trial_id):
        yield group


def trial_id(filename):
    '''
    >>> trial_id('1a2.xml')
    '1a2'
    >>> trial_id('people/1a3-0.xml')
    'people/1a3'
    >>> trial_id('1a4-0-6.txt')
    '1a4-0'
    >>> trial_id('plural.people.5.8.json')
    'plural.people.5.8'
    '''
    extension = filename.rfind('.')
    if extension != -1:
        filename = filename[:extension]
    hyphen = filename.rfind('-')
    if hyphen == -1 or not filename[hyphen + 1:].isdigit():
        return filename
    else:
        return filename[:hyphen]


class Trial:
    def __init__(self, filenames):
        self.filenames = filenames
        trees = [ET.parse(filename).getroot() for filename in filenames]
        # General trial-level attributes: cardinality, condition, domain, id
        for key, val in trees[0].attrib.items():
            try:
                val = int(val)
            except:
                pass
            setattr(self, key.lower(), val)
        # The set of entities in the <DOMAIN> element (there is always exactly 1):
        self.entities = [Entity(e) for e in trees[0][0].iter('ENTITY')]
        self.targets = [e for e in self.entities if e.is_target()]
        self.descriptions = [self.build_description(root) for root in trees]

    def build_description(self, root):
        # <STRING-DESCRIPTION> is always unique and the the 2nd daughter of the root:
        string_description_elem = root[1]
        # <DESCRIPTION> is always unique and the the 3rd daughter of the root:
        description_elem = root[2]
        # <ATTRIBUTE-SET> is always unique and the the 4th/final daughter of the root:
        attribute_set_elem = root[3]
        # More work needs to be done on descriptions if we want to use them beyond
        # string_description. For now, only string_description is fully configured.
        return Description(string_description_elem, description_elem, attribute_set_elem)

    def to_latex(self,
                 output_filename="temp.tex",
                 output_dirname=".",
                 img_dirname="TUNA/corpus/singular/furniture",
                 self_contained=True):
        xmax = self.xmax()
        ymax = self.ymax()
        # Framework for tabular environment:
        table = [["" for _ in range(xmax)] for _ in range(ymax)]
        for e in self.entities:
            # Covert the GIF to PNG, since pdflatex can't handle GIF:
            img_path = "%s/%s" % (img_dirname, e.image)
            png = "%s/%s" % (output_dirname, e.image.replace(".gif", ".png"))
            subprocess.Popen(["convert", img_path, png])
            # The cell has the image on left and uses a tabular environment to
            # list the attributes on the right:
            cell = ""
            cell += "\\parbox[c]{1.6cm}{\\includegraphics[scale=0.1]{%s}}\n" % png
            cell += "\\begin{tabular}{@{} c @{}}"
            x = None
            y = None
            attrs = e.attributes_as_dict()
            x = int(attrs['x-dimension'])-1
            y = int(attrs['y-dimension'])-1
            attr_strs = ["%s:%s" % key_val for key_val in sorted(attrs.items())]
            cell += "\\\\\n".join(attr_strs)
            cell += "\\end{tabular}"
            # Identify the target:
            if e.is_target():
                cell = "\colorbox{lightgray}{%s}" % cell
            else:
                cell = "\\framebox{%s}" % cell
            table[x][y] = cell
        # Format the cells as a table:
        tab = "\\begin{tabular}[c]{@{} *{%s}{c} @{}}\n" % ymax
        rows = [" & ".join(row) for row in table]
        rows = "\\\\\\\\\n".join(rows)
        tab += rows
        tab += "\\end{tabular}"
        tab = "\\framebox{\\scriptsize%s}" % tab
        # Format the utterance as a table:
        utt = "\\begin{tabular}{r l}"
        utt += "Utterance: &" + self.description.string_description + "\\\\\n"
        utt += " &  " + "; ".join([
            "%s:%s" % key_val
            for key_val in sorted(self.description.attributes_as_dict().items())
        ])
        utt += "\\end{tabular}"
        # The core content:
        s = tab + "\n\n" + utt
        # For a free-standing latexable file:
        if self_contained:
            s = r"\documentclass{article}\usepackage{colortbl}\usepackage{graphicx}" \
                r"\usepackage[usenames]{xcolor}\begin{document}\newcommand{\graycell}[1]" \
                r"{{\cellcolor[gray]{.8}#1}}\scriptsize" + s + r"\end{document}"
        # Output:
        if output_filename:
            open(output_filename, 'w').write(s)
        else:
            return s

    def dimmax(self, dim='x'):
        vals = [e.attributes_as_dict()['%s-dimension' % dim] for e in self.entities]
        try:
            return max([int(val) for val in vals])
        except:
            return None

    def xmax(self, dim='x'):
        return self.dimmax(dim='x')

    def ymax(self, dim='y'):
        return self.dimmax(dim='y')

######################################################################


class Entity:
    def __init__(self, element):
        # General entity-level attributes: id, image, type
        for key, val in element.attrib.items():
            try:
                val = int(val)
            except:
                pass
            setattr(self, key.lower(), val)
        self.attributes = [Attribute(e) for e in element.iter('ATTRIBUTE')]

    def is_target(self):
        return self.type == "target"

    def attributes_as_dict(self):
        return {a.name: a.value for a in self.attributes}

    def __eq__(self, e):
        """Defines equality in terms of equality of attributes"""
        if len(self.attributes) != len(e.attributes):
            return False
        return False not in [aself == a for aself, a in zip(self.attributes, e.attributes)]

    def __ne__(self, e):
        return not self.__eq__(e)

    def __contains__(self, a):
        for x in self.attributes:
            if x == a:
                return True
        return False


class Description:
    def __init__(self, string_description_elem, description_elem, attribute_set_elem):
        self.string_description = re.sub(r"\s*\n\s*", " ", string_description_elem.text.strip())
        self.description = description_elem
        self.attribute_set = [Attribute(a) for a in attribute_set_elem.iter('ATTRIBUTE')]

    def attributes_as_dict(self):
        return {a.name: a.value for a in self.attribute_set}

    def unigrams(self):
        return basic_unigram_tokenizer(self.string_description)


class Attribute:
    def __init__(self, element):
        self.type = None
        for key, val in element.attrib.items():
            setattr(self, key.lower(), val)

    def __str__(self):
        return ":".join([x for x in [self.name, self.value] if x])

    def __eq__(self, a):
        return (self.type == a.type) and (self.name == a.name) and (self.value == a.value)

    def __ne__(self, a):
        return not self.__eq__(a)

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, a):
        return str(self) < str(a)


if __name__ == '__main__':

    from collections import Counter
    from operator import itemgetter

    def stats():
        all_filenames = iglob("../TUNA/corpus/*/*/*.xml")
        corpus = TunaCorpus(all_filenames)
        counts = Counter([w for t in corpus.iter_trials() for w in t.description.unigrams()])
        for key, val in sorted(counts.items(), key=itemgetter(1), reverse=False):
            print key, val
        print
        print 'Vocab size:', len(counts)
        print 'Tokens:', sum(counts.values())

    import random
    import glob

    def find_display_candidates():
        filenames = glob.glob("../TUNA/corpus/singular/furniture/*.xml")
        random.shuffle(filenames)
        for filename in filenames:
            trial = Trial(filename)
            xvals = []
            yvals = []
            for e in trial.entities:
                attrs = e.attributes_as_dict()
                try:
                    xvals.append(int(attrs['x-dimension']))
                    yvals.append(int(attrs['y-dimension']))
                except:
                    break
            if xvals and yvals:
                xmax = max(xvals)
                ymax = max(yvals)
                if xmax <= 3 and ymax <= 5:
                    trial.to_latex()
                    break

    # find_display_candidates()

    # Example currently used in the paper:
    Trial("../TUNA/corpus/singular/furniture/s40t4.xml").to_latex()
