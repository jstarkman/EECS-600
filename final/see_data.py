#! /usr/bin/python2
import numpy as np
import tensorflow as tf
import png

import zxc251_reader

def depth2pixels(img):
	"Rescales into new array such that all values are between zero and one."
	a = np.amin(img)
	b = np.amax(img)
	s = 255.0 / (b - a)
	tmp = ((img - a) * s)
	return tmp.astype(int)

train_dataset, train_labels,\
	valid_dataset, valid_labels,\
	test_dataset, test_labels\
	= zxc251_reader.get_data()

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

first = train_dataset[0, :, :]
first_name = zxc251_reader.label2english(train_labels[0])
px = depth2pixels(first)
print px

# import pdb; pdb.set_trace() # set breakpoint

png.from_array(px, 'L;8').save("first_{}.png".format(first_name))

