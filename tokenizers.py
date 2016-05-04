import re

WORD_RE_STR = r"""
(?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
|
(?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
|
(?:[\w_]+)                     # Words without apostrophes or dashes.
|
(?:\.(?:\s*\.){1,})            # Ellipsis dots.
|
(?:\*{1,})                     # Asterisk runs.
|
(?:\S)                         # Everything else that isn't whitespace.
"""

WORD_RE = re.compile(r"(%s)" % WORD_RE_STR, re.VERBOSE | re.I | re.UNICODE)
 
def basic_unigram_tokenizer(s, lower=True):
    words = WORD_RE.findall(s)
    if lower:
        words = [w.lower() for w in words]
    return words
