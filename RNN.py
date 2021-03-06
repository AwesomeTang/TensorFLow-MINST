# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2019-02-22
# @version: python2.7


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os


class Config(object):
    """
    CNN 模型参数
    """
    classes = 10  # 类别数
    num_filters = 32  # 卷积核数
    kernel_size = 3  # 卷积核大小
    alpha = 1e-2  # 学习率
    keep_prob = 0.5  # 保留比例
    steps = 10000  # 迭代次数
    batch_size = 128  # 每批次训练样本数
    tensorboard_dir = 'tensorboard/RNN'  # log输出路径
    saver_folder = 'checkpoints_RNN'  # 模型保存路径
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    num_units = 128
    num_layers = 2

    decay_rate = 0.9  # 衰减率
    decay_steps = 100  # 衰减次数


class RNN:

    def __init__(self):
        print('Loading data......')
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.input_x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, Config.classes], name='input_y')
        self.keep_prob = tf.placeholder("float")

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

    def feed_data(self, x, y, keep_prob=1.):
        feed_dict = {self.input_x: x,
                     self.input_y: y,
                     self.keep_prob: keep_prob}
        return feed_dict

    def build_cells(self):
        """
        建立多层RNN网络
        :return: cells
        """
        cells = []
        for _ in range(Config.num_layers):
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=Config.num_units, name='basic_lstm_cell')
            rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell,
                                                     output_keep_prob=self.keep_prob)
            cells.append(rnn_cell)

        return cells

    def restore_model(self):
        """
        加载本地模型文件
        :return: session
        """
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=self.save_path)
        return session

    def rnn_model(self):
        # 定义RNN（LSTM）结构
        x_image = tf.reshape(self.input_x, [-1, 28, 28])
        cells = self.build_cells()
        rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                 inputs=x_image,
                                                 initial_state=None,
                                                 dtype=tf.float32,
                                                 time_major=False)
        output = tf.layers.dense(inputs=outputs[:, -1, :], units=Config.classes)

        gloabl_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(Config.alpha, gloabl_steps, Config.decay_steps,
                                                   Config.decay_rate, staircase=True)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=self.input_y, logits=output)  # 计算loss
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(loss, global_step=gloabl_steps)

        correct_prediction = tf.equal(tf.argmax(self.input_y, axis=1), tf.argmax(output, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))  # 计算正确率

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(Config.tensorboard_dir)

        return train_step, gloabl_steps, accuracy, loss, merged_summary, writer

    def run_model(self):
        train_step, gloabl_steps, accuracy, loss, merged_summary, writer = self.rnn_model()

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
            feed_dict = self.feed_data(data_x, data_y, Config.keep_prob)

            sess.run([train_step, gloabl_steps], feed_dict=feed_dict)

            if i % Config.save_per_batch == 0:
                feed_dict[self.keep_prob] = 1.
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
        print 'Time Usage : {:.2f} hours'.format(time_diff / 3600.)

        data_x, data_y = self.mnist.test.images, self.mnist.test.labels
        feed_dict = self.feed_data(data_x, data_y)
        sess = self.restore_model()
        test_acc, test_loss = sess.run([accuracy, loss],
                                       feed_dict=feed_dict)

        print("Test accuracy :{:8.2%}, loss:{:6.2f}".format(test_acc, test_loss))
        sess.close()


if __name__ == "__main__":
    RNN()
