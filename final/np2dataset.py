import numpy as np
import tensorflow as tf

def np2dataset(abcdef):
	"""Convert Numpy arrays to Tensorflow Dataset objects.

	Arguments:
	`abcdef`: a six-tuple of Numpy arrays in training-validation-testing order,
	with each of the three being in dataset-labels order.  See code if this is
	still unclear.
	"""
	train_dataset, train_labels,\
		valid_dataset, valid_labels,\
		test_dataset, test_labels\
		= abcdef
	return BucketDS( DS(train_dataset, train_labels),\
					 DS(valid_dataset, valid_labels),\
					 DS(test_dataset,  test_labels)  )

class BucketDS:
	def __init__(self, ds_train, ds_valid, ds_test):
		self.train = ds_train
		self.validation = ds_valid
		self.test = ds_test

class DS:
	def __init__(self, d, l):
		self.which_axis = d.shape.index(max(d.shape))
		# makes reshaping data into vectors relatively painless
		assert(self.which_axis == 0)
		self.num_examples = d.shape[self.which_axis]
		v = 1;
		for i, s in enumerate(d.shape):
			if i != self.which_axis:
				v *= s
		self.data   = d.reshape((self.num_examples, v))
		self.labels = l
		self.idx = 0
		self.epochs_completed = 0

	def next_batch(self, batch_size):
		start = self.idx
		self.idx += batch_size
		if self.idx > self.num_examples:
			self.epochs_completed += 1
			self.shuffle()
			start = 0
			self.idx = batch_size
		end = self.idx
		final = slice(start, end)
		sl = [slice(None, None)] * len(self.data.shape)
		sl[self.which_axis] = final
		return self.data[tuple(sl)], self.labels[final]
		# return self.data[start:end], self.labels[start:end]

	def shuffle(self):
		p = np.arange(self.num_examples)
		np.random.shuffle(p)
		self.data = self.data[p] # copy b/c Numpy advanced indexing
		self.labels = self.labels[p] # same

