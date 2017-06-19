#! /usr/bin/python3
from six.moves import cPickle as pickle
import numpy as np

file_name = 'depth_data.pickle4'
new_fn = 'depth_data.pickle2'
with open(file_name, 'rb') as f, open(new_fn, 'wb') as out:
	save = pickle.load(f)
	pickle.dump(save, out, protocol=2)
	# dataset = save['dataset']
	# names = save['names']
	# orientations = save['orientations']

