#!/usr/bin/env python3

import re

text = """\'hallo, \t  dit is tekst.\n    \'"""

print(repr("text: {}".format(text)))
print("text: {}".format(text))
r = text.rstrip('\t')
print("rstrip: {}".format(r))
t_re = re.sub(r"\s+", ' ', text)
print("re sub: {}".format(t_re))
bla = t_re.lstrip()
print(repr("re sub + rstrip: {}".format(bla)))