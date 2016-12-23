#! /usr/local/src/RPMS/anaconda2/bin/python2
import numpy as np
import caffe

import zxc251_reader

train_dataset, train_labels,\
	valid_dataset, valid_labels,\
	test_dataset, test_labels\
	= zxc251_reader.get_data()

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

