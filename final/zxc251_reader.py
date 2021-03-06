#! /usr/local/src/RPMS/anaconda2/bin/python2
import numpy as np
from six.moves import cPickle as pickle


name2value = {'v8':0, 'duck':1, 'stapler':2, 'pball':3, 'tball':4, 'sponge':5,
			  'bclip':6, 'tape':7, 'gstick':8, 'cup':9, 'pen':10, 'calc':11,
			  'blade':12, 'bottle':13, 'cpin':14, 'scissors':15, 'stape':16,
			  'gball':17, 'orwidg':18, 'glue':19, 'spoon':20, 'fork':21,
			  'nerf':22, 'eraser':23, 'empty':24}

name2string = {'v8':'v8 can', 'duck':'ducky', 'stapler':'stapler',
			   'pball':'ping pong ball', 'tball':'tennis ball', 'sponge':'sponge',
			   'bclip':'binder clip', 'tape':'big tape', 'gstick':'glue stick',
			   'cup':'cup', 'pen':'pen', 'calc':'calculator', 'blade':'razor',
			   'bottle':'bottle', 'cpin':'clothespin', 'scissors':'scissors',
			   'stape':'small tape', 'gball':'golf ball', 'orwidg':'orange thing',
			   'glue':'glue', 'spoon':'spoon', 'fork':'fork',
			   'nerf':'nerf gun', 'eraser':'eraser', 'empty':'empty plate'}

value2name = dict((value, name) for name, value in name2value.items())

def label2english(l):
	return name2string[value2name[l]]

def get_data():
	file_name = 'depth_data.pickle2'
	with open(file_name, 'rb') as f:
		save = pickle.load(f)
		dataset = save['dataset']
		names = save['names']
		orientations = save['orientations']
		del save

	# generate labels
	# for 10 objectives
	image_size = 34
	num_labels = 25
	num_channels = 1

	num_images = dataset.shape[0]
	num_train = int(round(num_images*0.85))
	num_valid = int(round(num_images*0.08))
	num_test = int(round(num_images*0.1))

	labels = np.ndarray(num_images, dtype=np.int32)
	index = 0
	for name in names:
		labels[index] = name2value[name]
		index += 1

	def randomize(dataset, labels):
		permutation = np.random.permutation(labels.shape[0])
		shuffled_dataset = dataset[permutation, :, :]
		shuffled_labels = labels[permutation]
		return shuffled_dataset, shuffled_labels

	rdataset, rlabels = randomize(dataset[:, :, :, np.newaxis], labels)
	train_dataset = rdataset[0:num_train, :, :, :]
	train_labels = rlabels[0:num_train]
	valid_dataset = rdataset[num_train:(num_train+num_valid), :, :, :]
	valid_labels = rlabels[num_train:(num_train+num_valid)]
	test_dataset = rdataset[(num_train+num_valid):, :, :, :]
	test_labels = rlabels[(num_train+num_valid):]
	# print('Training:', train_dataset.shape, train_labels.shape)
	# print('Validation:', valid_dataset.shape, valid_labels.shape)
	# print('Testing:', test_dataset.shape, test_labels.shape)
	return (train_dataset, train_labels,\
			valid_dataset, valid_labels,\
			test_dataset, test_labels\
	)
