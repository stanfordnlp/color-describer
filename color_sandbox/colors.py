#!/usr/bin/env python
import sqlite3
import json
from collections import namedtuple

User = namedtuple('User',
  'id, user_key, datestamp, ip, language, monitor, temperature, gamma, '
  'colorblind, ychrom, samplecolors, spamprob'
)
Answer = namedtuple('Answer',
  'id, user_id, datestamp, r, g, b, colorname'
)
Name = namedtuple('Name',
  'id, colorname, numusers, numinstances'
)


def load_table(name, objtype):
  filename = '/scr/nlp/data/xkcdcolors/%s.json' % name
  with open(filename, 'r') as infile:
    map = json.load(infile)
  return {k: objtype(*v) for k, v in map.iteritems()}


USERS = load_table('users', User)
ANSWERS = load_table('answers', Answer)
NAMES = load_table('names', Name)
