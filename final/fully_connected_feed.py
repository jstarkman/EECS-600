#! /usr/bin/python

"""Trains and Evaluates the KDD network using a feed dictionary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.examples.tutorials.mnist import mnist
import zxc251_reader
import kdd
from np2dataset import np2dataset

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')

# kdd.inference(images, c1, stack_height_c1, c2, stack_height_c2, fc1, fc2):
flags.DEFINE_integer('conv1',    5, 'Side length of kernel of conv1.')
flags.DEFINE_integer('stack1',  64, 'Number of feature maps for conv1.')
flags.DEFINE_integer('conv2',    5, 'Side length of kernel of conv2.')
flags.DEFINE_integer('stack2', 128, 'Number of feature maps for conv2.')
flags.DEFINE_integer('fc1',    192, 'Number of units in fully-connected layer 1.')
flags.DEFINE_integer('fc2',     96, 'Number of units in fully-connected layer 2.')

flags.DEFINE_integer('batch_size', 243, 'Batch size.  Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

def fill_feed_dict(data_set, images_pl, labels_pl):
	"""Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
	"""
	images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
	feed_dict = {
		images_pl: images_feed,
		labels_pl: labels_feed,
	}
	return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
	"""Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
	"""
	true_count = 0 # Counts the number of correct predictions.
	steps_per_epoch = data_set.num_examples // FLAGS.batch_size
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
		true_count += sess.run(eval_correct, feed_dict=feed_dict)
	
	precision = true_count / dataset.num_examples
	print('	Num examples: %d | Num correct: %d | Precision @ 1: %0.04f' %
		  (num_examples, true_count, precision))
	return precision

def run_training():
	"""Train KDD for a number of steps."""
	# (train_dataset, train_labels,\
	#  valid_dataset, valid_labels,\
	#  test_dataset, test_labels) = zxc251_reader.get_data()

	with tf.Graph().as_default():
		d = zxc251_reader.get_data()
		ds = []
		for i, t in enumerate(d):
			if len(t.shape) > 1: # if not labels
				t = t[:, :, :, np.newaxis]
			ds.append(tf.convert_to_tensor(t))

		global_step = tf.contrib.framework.get_or_create_global_step()

		images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, kdd.IMAGE_SIZE, kdd.IMAGE_SIZE, 1))
		labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

		logits = kdd.inference(images_placeholder, FLAGS.conv1, FLAGS.stack1, FLAGS.conv2,
							   FLAGS.stack2, FLAGS.fc1, FLAGS.fc2)
		loss = kdd.loss(logits, labels_placeholder)
		train_op = kdd.training(loss, global_step, FLAGS.batch_size)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		sess = tf.Session()
		sess.run(init)
		t_start = time.time()
		for step in xrange(FLAGS.max_steps):
			feed_dict = fill_feed_dict(train_dataset, images_placeholder, labels_placeholder)
			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			if step % 100: # status report
				print("Step {}, loss = {}, time so far = {}"
					  .format(step, loss_value, time.time() - t_start))
				if (step + 1) % 1000 in {0, FLAG.max_steps}:
					filename_checkpoint = os.path.join(FLAG.train_dir, "model.checkpoint")
					saver.save(sess, filename_checkpoint, global_step=global_step)
					

		# with tf.train.MonitoredTrainingSession( # Jennings lab only has TF v0.10rc0
		# 		checkpoint_dir=FLAGS.train_dir,
		# 		hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
		# 			   tf.train.NanTensorHook(loss) ]) as mon_sess:
		# 	while not mon_sess.should_stop():
		# 		mon_sess.run(train_op)

def main(argv=None):
	run_training()

if __name__ == '__main__':
	tf.app.run()
