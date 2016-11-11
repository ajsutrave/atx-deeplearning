# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function


import argparse
import cv2
import numpy as np

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # img = np.reshape(mnist.test.images[0], (28, 28))
  # cv2.imshow('img', img)
  # cv2.waitKey(0)
  # cv2.imwrite('mnist.test.images0.jpg', img*256)
  # img_vector = np.reshape(img, np.shape(mnist.test.images[0]))
  # # print sess.run(y, feed_dict={x: [mnist.test.images[0]]})
  # # print sess.run(tf.argmax(y, 1), feed_dict={x: [img_vector]})

  if FLAGS.img:
    img = np.ones((28, 28)) - (cv2.imread(FLAGS.img, cv2.CV_LOAD_IMAGE_GRAYSCALE) / 256.)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    img_vector = np.reshape(img, np.shape(mnist.test.images[0]))
    print sess.run(tf.nn.softmax(y), feed_dict={x: [img_vector]})
    print sess.run(tf.argmax(y, 1), feed_dict={x: [img_vector]})

  #   # # cv2.waitKey(0)
  #   # # img_vector = np.reshape(img, np.shape(mnist.test.images[0]))
  #   # img_vector = np.reshape(img, np.shape(mnist.test.images[0]))
  #   # # print (np.shape(mnist.test.images[0]), np.shape(img_vector))
  #   # # print (mnist.test.images[0])
  #   # # print (img_vector)
  #   # y_ = [0] * 10; y_[4] = 1
  #   # print sess.run(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)), feed_dict={x: [img_vector]})
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',help='Directory for storing data')
  parser.add_argument('--img', type=str, help='Image to test against')
  FLAGS = parser.parse_args()
  tf.app.run()
