#! /usr/bin/python

"""Trains and Evaluates the KDD network using a feed dictionary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import six
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

def do_eval(sess, qty_correct_answers, feed_dict):
	"""Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    qty_correct_answers: The Tensor that returns the number of correct predictions.
    feed_dict: Holding any of training, validation, or testing data.
	"""
	true_count = 0 # Counts the number of correct predictions.
	qty = six.next(six.itervalues(feed_dict)).shape[0]
	steps_per_epoch = qty // FLAGS.batch_size
	for step in xrange(steps_per_epoch):
		true_count += sess.run(qty_correct_answers, feed_dict=feed_dict)
	
	precision = true_count / qty
	print('	Num examples: %d | Num correct: %d | Precision @ 1: %0.04f' %
		  (qty, true_count, precision))

def run_training():
	"""Train KDD for a number of steps."""
	with tf.Graph().as_default():
		ds = np2dataset(zxc251_reader.get_data()) # returns np arrays

		global_step = tf.contrib.framework.get_or_create_global_step()

		images_placeholder = tf.placeholder(
			tf.float32, shape=(FLAGS.batch_size, kdd.IMAGE_SIZE, kdd.IMAGE_SIZE, 1))
		labels_placeholder = tf.placeholder(
			tf.int32, shape=(FLAGS.batch_size,))
		def make_feed_dict(dataset):
			nb = dataset.next_batch(FLAGS.batch_size)
			return { images_placeholder: nb[0], labels_placeholder: nb[1] }

		logits = kdd.inference(images_placeholder, FLAGS.conv1, FLAGS.stack1, FLAGS.conv2,
							   FLAGS.stack2, FLAGS.fc1, FLAGS.fc2)
		loss = kdd.loss(logits, labels_placeholder)
		train_op = kdd.training(loss, global_step, FLAGS.batch_size)
		qty_correct_answers = kdd.evaluation(logits, labels_placeholder)
		if tf.__version__ == "0.10.0rc0": # Jennings lab
			init = tf.initialize_all_variables()
		else: # at least as of v0.12.0
			init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		sess = tf.Session()
		sess.run(init)
		t_start = time.time()
		for step in xrange(FLAGS.max_steps):
			feed_dict = make_feed_dict(ds.train)
			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			if step % 100 in {0, FLAGS.max_steps}: # status report
				print("Step {}, loss = {}, time so far = {}"
					  .format(step, loss_value, time.time() - t_start))
			if (step + 1) % 1000 in {0, FLAGS.max_steps}:
				filename_checkpoint = os.path.join(FLAGS.train_dir, "model.checkpoint")
				saver.save(sess, filename_checkpoint, global_step=global_step)
				do_eval(sess, qty_correct_answers, make_feed_dict(ds.train))
				do_eval(sess, qty_correct_answers, make_feed_dict(ds.validation))
				do_eval(sess, qty_correct_answers, make_feed_dict(ds.test))

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
