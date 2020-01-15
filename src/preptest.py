#!/usr/bin/env python
import sys, traceback

t = 0
for line in sys.stdin:
	try:
		t += 1
		#sys.stdout.write("{}".format(line))
		if ',' in line:
			c,x = line.split(',')
			sys.stdout.write('{},{}\n'.format(c, repr(x)))
		else:
			raise Exception()
	except:
		sys.stdout.write("ERROR (r.{}): {}\n".format(t, line))

sys.stdout.write('\n\nnumber of lines: {}\n'.format(t))