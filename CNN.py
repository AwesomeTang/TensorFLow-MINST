# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2018-12-15
# @version: python2.7


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Config(object):
    """
    CNN 模型参数
    """
    classes = 10  # 类别数
    num_filters = 32  # 卷积核数
    kernel_size = 3  # 卷积核大小
    alpha = 1e-3  # 学习率
    keep_prob = 0.5  # 保留比例
    steps = 100000  # 迭代次数
    batch_size = 128  # 每批次训练样本数
    tensorboard_dir = 'tensorboard/CNN'  # log输出路径
    saver_folder = 'checkpoints'  # 模型保存路径
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    decay_rate = 0.8  # 衰减率
    decay_steps = 1000  # 衰减次数


class CNN:

    def __init__(self):
        print('Loading data......')
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.input_x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, Config.classes], name='input_y')
        self.keep_prob = tf.placeholder("float")
        self.is_training = tf.placeholder(tf.bool)

        if not os.path.exists(Config.saver_folder):
            os.mkdir(Config.saver_folder)
        self.save_path = os.path.join(Config.saver_folder, 'best_validation')

        self.run_model()

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

    def feed_data(self, x, y, keep_prob=1.0, is_training=False):
        feed_dict = {self.input_x: x,
                     self.input_y: y,
                     self.keep_prob: keep_prob,
                     self.is_training: is_training}
        return feed_dict

    def restore_model(self):
        """
        读取保存的模型
        用于评估测试集准确率
        :return: session
        """
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=self.save_path)
        return session

    def cnn_model(self):
        x_image = tf.reshape(self.input_x, [-1, 28, 28, 1])

        """
        第一层： 卷积
        (batch_size, 28, 28, 1)->(batch_size, 14, 14, 32)
        """
        w_cv1 = self.weight_variable([5, 5, 1, 32])
        b_cv1 = self.bias_variable([32])
        h_cv1 = tf.nn.relu(tf.add(self.conv2d(x_image, w_cv1), b_cv1))
        h_mp1 = self.max_pool_2x2(h_cv1)
        h_nm1 = tf.layers.batch_normalization(h_mp1, training=self.is_training, momentum=0.9)
        h_mp1 = tf.nn.dropout(h_nm1, self.keep_prob)

        """
        第二层： 卷积
        (batch_size, 14, 14, 32)->(batch_size, 7, 7, 64)
        """
        w_cv2 = self.weight_variable([5, 5, 32, 64])
        b_cv2 = self.bias_variable([64])
        h_cv2 = tf.nn.relu(tf.add(self.conv2d(h_mp1, w_cv2), b_cv2))
        h_mp2 = tf.nn.relu(self.max_pool_2x2(h_cv2))
        h_nm2 = tf.layers.batch_normalization(h_mp2, training=self.is_training, momentum=0.9)
        h_mp2 = tf.nn.dropout(h_nm2, self.keep_prob)

        """
        第三层： 全连接
        (batch_size, 7, 7, 64)->(batch_size, 7 * 7 * 64)->(batch_size, 128)
        """
        w_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_mp2_flat = tf.reshape(h_mp2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_mp2_flat, w_fc1), b_fc1))
        h_nm3 = tf.layers.batch_normalization(h_fc1, training=self.is_training, momentum=0.9)
        h_fc1 = tf.nn.dropout(h_nm3, self.keep_prob)

        """
        第四层： 全连接
        (batch_size, 128)->(batch_size, 10)
        """
        w_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        y_conv = tf.add(tf.matmul(h_fc1, w_fc2), b_fc2)

        # 变换学习率
        gloabl_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(Config.alpha, gloabl_steps, Config.decay_steps,
                                                   Config.decay_rate, staircase=True)

        # Adam优化器
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=y_conv))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(
                learning_rate).minimize(loss, global_step=gloabl_steps)

        # 计算准确率
        correct_prediction = tf.equal(
            tf.argmax(y_conv, 1), tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Tensor Board配置
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(Config.tensorboard_dir)
        return train_step, gloabl_steps, accuracy, loss, merged_summary, writer

    def run_model(self):
        train_step, gloabl_steps, accuracy, loss, merged_summary, writer = self.cnn_model()

        saver = tf.train.Saver(max_to_keep=1)

        start_time = datetime.now()
        best_acc = 0
        last_improved_step = 0
        require_steps = 10000

        print('Training & Evaluating......')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(Config.steps):
            data_x, data_y = self.mnist.train.next_batch(Config.batch_size)
            feed_dict = self.feed_data(data_x, data_y, Config.keep_prob, True)

            sess.run([train_step, gloabl_steps], feed_dict=feed_dict)

            if i % Config.save_per_batch == 0:
                feed_dict[self.keep_prob] = 1.
                feed_dict[self.is_training] = False
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, i)

            if i % Config.print_per_batch == 0:
                train_acc, train_loss = sess.run([accuracy, loss],
                                                 feed_dict=feed_dict)
                data_x, data_y = self.mnist.validation.images, self.mnist.validation.labels
                feed_dict = self.feed_data(data_x, data_y)
                val_acc, val_loss = sess.run([accuracy, loss],
                                             feed_dict=feed_dict)
                if val_acc > best_acc:
                    # 只在验证集准确率有提升的时候保存模型
                    best_acc = val_acc
                    last_improved_step = i
                    saver.save(sess=sess, save_path=self.save_path)
                    improved = '*'
                else:
                    improved = ''

                msg = 'Step {:5}, train_acc:{:8.2%}, train_loss:{:6.2f},' \
                      ' val_acc:{:8.2%}, val_loss:{:6.2f}, improved:{:3}'
                print(msg.format(i, train_acc, train_loss, val_acc, val_loss, improved))

            if i - last_improved_step > require_steps:
                # 超过require_steps验证集准确率提升，提前结束训练
                print('No improvement for over %d steps, auto-stopping....' % require_steps)
                break

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print('Time Usage : {:.2f} hours'.format(time_diff / 3600.))

        # 输出测试集准确率
        data_x, data_y = self.mnist.test.images, self.mnist.test.labels
        feed_dict = self.feed_data(data_x, data_y)

        # 加载本地模型文件
        sess = self.restore_model()
        test_acc, test_loss = sess.run([accuracy, loss],
                                       feed_dict=feed_dict)

        print("Test accuracy :{:8.2%}, loss:{:6.2f}".format(test_acc, test_loss))
        sess.close()


if __name__ == "__main__":
    CNN()
