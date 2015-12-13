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

USERS = {}
ANSWERS = {}
NAMES = {}


def users(id, *args):
  USERS[id] = User(id, *args)


def answers(id, *args):
  ANSWERS[id] = Answer(id, *args)


def names(id, *args):
  NAMES[id] = Name(id, *args)


conn = sqlite3.connect('/scr/nlp/data/xkcdcolors/db.sqlite')
c = conn.cursor()

for table in [users, answers, names]:
  for row in c.execute('SELECT * from %s' % table.__name__):
    table(*row)

conn.close()

if __name__ == '__main__':
  for table in 'users', 'answers', 'names':
    print table
    obj = locals()[table.upper()]
    with open(table + '.json', 'w') as outfile:
      json.dump(obj, outfile)
