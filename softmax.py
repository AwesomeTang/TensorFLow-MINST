# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2018-12-16
# @version: python2.7

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime


class Constant(object):
    classes = 10  # 类别数
    alpha = 1e-2  # 学习率
    steps = 10000  # 迭代次数
    batch_size = 50  # 每批次训练样本数
    print_per_batch = 100  # 每多少轮输出一次结果
    tensorboard_dir = 'tensorboard/softmax'
    save_per_batch = 10


class SoftMax:

    def __init__(self, constant):
        self.constant = constant
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.input_x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, self.constant.classes], name='input_y')

        self.run_model()

    def feed_data(self, x, y):
        feed_dict = {self.input_x: x,
                     self.input_y: y}
        return feed_dict

    def run_model(self):

        weight = tf.Variable(tf.zeros([784, 10]))
        bias = tf.Variable(tf.zeros([10]))

        y = tf.add(tf.matmul(self.input_x, weight), bias)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=y))
        train_step = tf.train.GradientDescentOptimizer(
            self.constant.alpha).minimize(cross_entropy)
        correct_prediction = tf.equal(
            tf.argmax(y, 1), tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        tf.summary.scalar("loss", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.constant.tensorboard_dir)

        start_time = datetime.now()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(self.constant.steps):
            data_x, data_y = self.mnist.train.next_batch(self.constant.batch_size)
            feed_dict = self.feed_data(data_x, data_y)
            sess.run(train_step, feed_dict=feed_dict)

            if i % self.constant.save_per_batch == 0:
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, i)

            if i % self.constant.print_per_batch == 0:
                train_acc, train_loss = sess.run([accuracy, cross_entropy],
                                                 feed_dict=feed_dict)
                data_x, data_y = self.mnist.validation.images, self.mnist.validation.labels
                feed_dict = self.feed_data(data_x, data_y)
                val_acc, val_loss = sess.run([accuracy, cross_entropy],
                                             feed_dict=feed_dict)
                msg = 'Step {:5}, train_acc:{:8.2%}, train_loss:{:6.2f},' \
                      ' val_acc:{:8.2%}, val_loss:{:6.2f}'
                print msg.format(i, train_acc, train_loss, val_acc, val_loss)

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print 'Time Usage : {:.2f} mins'.format(time_diff / 60.)

        data_x, data_y = self.mnist.test.images, self.mnist.test.labels
        feed_dict = self.feed_data(data_x, data_y)
        test_acc, test_loss = sess.run([accuracy, cross_entropy],
                                       feed_dict=feed_dict)

        print "Test accuracy :{:8.2%}, loss:{:6.2f}".format(test_acc, test_loss)
        sess.close()


if __name__ == "__main__":
    SoftMax(Constant)
