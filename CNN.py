# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2018-12-15
# @version: python2.7



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime


class constant(object):
    """
    CNN 模型参数
    """
    classes = 10  # 类别数
    num_filters = 32  # 卷积核数
    kernel_size = 3  # 卷积核大小
    alpha = 1e-3  # 学习率
    keep_prob = 0.5  # 保留比例
    steps = 10000  # 迭代次数
    batch_size = 50  # 每批次训练样本数
    tensorboard_dir = 'tensorboard/CNN'  # log输出路径
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    decay_rate = 0.95  # 衰减率
    decay_steps = 100  # 衰减次数


class CNN:

    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.input_x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, constant.classes], name='input_y')
        self.keep_prob = tf.placeholder("float")

        self.cnn_model()

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def feed_data(self, x, y, keep_prob=1.):
        feed_dict = {self.input_x: x,
                     self.input_y: y,
                     self.keep_prob: keep_prob}
        return feed_dict

    def cnn_model(self):
        #  第一层： 卷积
        x_image = tf.reshape(self.input_x, [-1, 28, 28, 1])
        w_cv1 = self.weight_variable([3, 3, 1, 32])
        b_cv1 = self.bias_variable([32])
        h_cv1 = tf.nn.relu(self.conv2d(x_image, w_cv1) + b_cv1)
        h_mp1 = self.max_pool_2x2(h_cv1)

        # 第二层： 卷积
        w_cv2 = self.weight_variable([3, 3, 32, 64])
        b_cv2 = self.bias_variable([64])
        h_cv2 = tf.nn.relu(self.conv2d(h_mp1, w_cv2) + b_cv2)
        h_mp2 = self.max_pool_2x2(h_cv2)

        # 第三层： 全连接
        w_fc1 = self.weight_variable([7 * 7 * 64, 128])
        b_fc1 = self.bias_variable([128])
        h_mp2_flat = tf.reshape(h_mp2, [-1, 7 * 7 * 64])
        h_fc1 = tf.add(tf.matmul(h_mp2_flat, w_fc1), b_fc1)

        # 第四层： Dropout层

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # 第五层： softmax输出层
        w_fc2 = self.weight_variable([128, 10])
        b_fc2 = self.bias_variable([10])
        y_conv = tf.nn.relu(tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2))

        gloabl_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(constant.alpha, gloabl_steps, constant.decay_steps,
                                                   constant.decay_rate, staircase=True)

        #
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=y_conv))
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(cross_entropy, global_step=gloabl_steps)
        correct_prediction = tf.equal(
            tf.argmax(y_conv, 1), tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        loss = tf.reduce_mean(cross_entropy)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(constant.tensorboard_dir)

        start_time = datetime.now()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(constant.steps):
            data_x, data_y = self.mnist.train.next_batch(constant.batch_size)
            feed_dict = self.feed_data(data_x, data_y, constant.keep_prob)

            sess.run([train_step,gloabl_steps], feed_dict= feed_dict)
            if i % constant.save_per_batch == 0:
                feed_dict[self.keep_prob] = 1.
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, i)

            if i % constant.print_per_batch == 0:
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
        print 'Time Usage : {:.2f} hours'.format(time_diff / 3600.)

        data_x, data_y = self.mnist.test.images, self.mnist.test.labels
        feed_dict = self.feed_data(data_x, data_y)
        test_acc, test_loss = sess.run([accuracy, cross_entropy],
                                       feed_dict=feed_dict)

        print "Test accuracy :{:8.2%}, loss:{:6.2f}".format(test_acc, test_loss)
        sess.close()


if __name__ == "__main__":
    CNN()
