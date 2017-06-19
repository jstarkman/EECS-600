"""Builds the Kinect Depth Data network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.

Liberally borrowed from here:
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

NUM_CLASSES = 25
IMAGE_SIZE = 34
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 41310
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000 # same as CIFAR-10

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

def get_lossy_variable(name, shape, stddev, wd):
	var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
	weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
	tf.add_to_collection('losses', weight_decay)
	return var

def layer_conv(x, kernel_shape):
	"""Simple convolution layer with nonlinearity.  All HP guesses.
	Arguments:
	`x`: input tensor, shape = [batchLength, height, width, channels]
	`kernel_shape`: [height, width, channelsInput, channelsOutput]
	Reference: https://www.tensorflow.org/api_docs/python/nn/convolution#conv2d
	"""
	# bias_shape = [kernel_shape[-1]]
	# bias = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
	# outp = tf.nn.relu(conv + biases)
	wait = get_lossy_variable("weights", kernel_shape, 0.05, 0.0)
	conv = tf.nn.conv2d(x, wait, strides=([1]*4), padding="SAME")
	pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
	norm = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')
	return norm

def layer_fc(x, qty_outputs, stddev=0.04, wd=0.004):
	"""Simple fully-connected layer with nonlinearity.  All HP guesses.
	Arguments:
	`x`: input tensor, shape = [batchLength, height, width, channels]
	`qty_outputs`: length of next vector in chain
	"""
	dims = x.get_shape()[1].value
	wait = get_lossy_variable('weights', shape=[dims, qty_outputs], stddev=stddev, wd=wd)
	bias = tf.get_variable('biases', [qty_outputs], initializer=tf.constant_initializer(0.1))
	outp = tf.nn.relu(tf.matmul(x, wait) + bias, name="ReLU")
	return outp

def inference(images, c1, stack_height_c1, c2, stack_height_c2, fc1, fc2):
	p = images

	with tf.variable_scope('conv1') as scope:
		p = layer_conv(p, [c1, c1, 1, stack_height_c1])

	with tf.variable_scope('conv2') as scope:
		p = layer_conv(p, [c2, c2, stack_height_c1, stack_height_c2])

	p = tf.reshape(p, [int(p.get_shape()[0]), -1]) # images -> vector

	with tf.variable_scope("fc1") as scope:
		p = layer_fc(p, fc1)

	with tf.variable_scope("fc2") as scope:
		p = layer_fc(p, fc2)

	with tf.variable_scope("fc3") as scope:
		p = layer_fc(p, NUM_CLASSES, stddev=(1/NUM_CLASSES), wd=0.0)

	# no softmax because loss() will do it for us

	return p

def loss(logits, labels):
	"""Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns: Loss tensor of type float.
	"""
	labels = tf.to_int64(labels)
	# http://stackoverflow.com/a/37317322
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	tf.add_to_collection("losses", cross_entropy_mean)
	return tf.add_n(tf.get_collection("losses"), name="total_loss")

def training(loss, global_step, batch_size):
	"""Sets up the training Ops.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    global_step: int32 Variable counting how many steps we have taken so far

  Returns:
    train_op: The Op for training.
	"""
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
	# https://www.tensorflow.org/api_docs/python/train/decaying_the_learning_rate#exponential_decay
	lr = tf.train.exponential_decay(\
		INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

	# generate the moving averages of all of the losses
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [loss])

	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(loss)

	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op

def evaluation(logits, labels):
	"""Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
	"""
	correct = tf.nn.in_top_k(logits, labels, 1)
	# Return the number of true entries.
	return tf.reduce_sum(tf.cast(correct, tf.int32))

