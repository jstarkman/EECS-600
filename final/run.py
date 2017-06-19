#!/usr/bin/python
import subprocess

# flags.DEFINE_integer('conv1',    5, 'Side length of kernel of conv1.')
# flags.DEFINE_integer('stack1',  64, 'Number of feature maps for conv1.')
# flags.DEFINE_integer('conv2',    5, 'Side length of kernel of conv2.')
# flags.DEFINE_integer('stack2', 128, 'Number of feature maps for conv2.')
# flags.DEFINE_integer('fc1',    192, 'Number of units in fully-connected layer 1.')
# flags.DEFINE_integer('fc2',     96, 'Number of units in fully-connected layer 2.')

def do_iter(args):
	fn = "log_" + "-".join(str(a) for a in args) + ".txt"
	print("  About to do this combination: " + fn)
	with open(fn, "w") as f:
		pass_me = "--conv1 {} --stack1 {} --conv2 {} --stack2 {} --fc1 {} --fc2 {}".format(*args)
		print pass_me
		subprocess.call(["./fully_connected_feed.py", pass_me], stdout=f, stderr=subprocess.STDOUT)

conv1 = [3, 5]
stack1 = [32, 64]
conv2 = [3, 5]
stack2 = [64, 128]
fc1 = [96, 192]
fc2 = [48, 96]

for c1 in conv1:
	for s1 in stack1:
		for c2 in conv2:
			for s2 in stack2:
				for f1 in fc1:
					for f2 in fc2:
						do_iter([c1, s2, c2, s2, f1, f2])

