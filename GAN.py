# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2019-02-22
# @version: python2.7


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Config:
    alpha = 1e-2
    drop_rate = 0.5  # 保留比例
    steps = 300000  # 迭代次数
    batch_size = 128  # 每批次训练样本数
    epochs = 100  # 训练轮次

    num_units = 128
    size = 784
    noise_size = 100

    smooth = 0.01
    learning_rate = 1e-4

    print_per_step = 1000


class Gan:

    def __init__(self):
        print('Loading data......')
        # 读取MNIST数据集
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # 定义占位符，真实图片和生成的图片
        self.real_images = tf.placeholder(tf.float32, [None, Config.size], name='real_images')
        self.noise = tf.placeholder(tf.float32, [None, Config.noise_size], name='noise')
        self.drop_rate = tf.placeholder('float')

        self.train_step()

    def generator_graph(self, noise, n_units, out_dim, alpha, reuse=False):
        # todo 生成器
        with tf.variable_scope('generator', reuse=reuse):
            # Hidden layer
            h1 = tf.layers.dense(noise, n_units, activation=None)
            # Leaky ReLU
            h1 = tf.maximum(alpha * h1, h1)
            h1 = tf.layers.dropout(h1, rate=self.drop_rate)
            # Logits and tanh output
            logits = tf.layers.dense(h1, out_dim, activation=None)
            out = tf.tanh(logits)

        return out

    @staticmethod
    def discriminator_graph(image, n_units, alpha, reuse=False):
        # todo 判别器
        with tf.variable_scope('discriminator', reuse=reuse):
            # Hidden layer
            h1 = tf.layers.dense(image, n_units, activation=None)
            # Leaky ReLU
            h1 = tf.maximum(alpha * h1, h1)

            logits = tf.layers.dense(h1, 1, activation=None)
            # out = tf.sigmoid(logits)

        return logits

    def net(self):
        # generator
        fake_image = self.generator_graph(self.noise, Config.num_units, Config.size, Config.alpha)

        # discriminator
        real_logits = self.discriminator_graph(self.real_images, Config.num_units, Config.alpha)
        fake_logits = self.discriminator_graph(fake_image, Config.num_units, Config.alpha, reuse=True)

        # discriminator的loss
        # 识别真实图片
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)) * (
                    1 - Config.smooth))
        # 识别生成的图片
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        # 总体loss
        d_loss = tf.add(d_loss_real, d_loss_fake)

        # generator的loss
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)) * (
                    1 - Config.smooth))

        net_vars = tf.trainable_variables()

        # generator中的tensor
        g_vars = [var for var in net_vars if var.name.startswith("generator")]
        # discriminator中的tensor
        d_vars = [var for var in net_vars if var.name.startswith("discriminator")]

        # optimizer
        dis_optimizer = tf.train.AdamOptimizer(Config.learning_rate).minimize(d_loss, var_list=d_vars)
        gen_optimizer = tf.train.AdamOptimizer(Config.learning_rate).minimize(g_loss, var_list=g_vars)

        return dis_optimizer, gen_optimizer, d_loss, g_loss

    def train_step(self):
        dis_optimizer, gen_optimizer, d_loss, g_loss = self.net()

        print('Training & Evaluating......')
        start_time = datetime.now()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(Config.steps):
            real_image, _ = self.mnist.train.next_batch(Config.batch_size)

            real_image = real_image * 2 - 1

            # generator的输入噪声
            batch_noise = np.random.uniform(-1, 1, size=(Config.batch_size, Config.noise_size))

            sess.run(gen_optimizer, feed_dict={self.noise: batch_noise, self.drop_rate: Config.drop_rate})
            sess.run(dis_optimizer, feed_dict={self.noise: batch_noise, self.real_images: real_image})

            if step % Config.print_per_step == 0:
                dis_loss = sess.run(d_loss, feed_dict={self.noise: batch_noise, self.real_images: real_image})
                gen_loss = sess.run(g_loss, feed_dict={self.noise: batch_noise, self.drop_rate: 1.})
                end_time = datetime.now()
                time_diff = (end_time - start_time).seconds

                msg = 'Step {:3} k Dis_Loss:{:6.2f}, Gen_Loss:{:6.2f}, Time_Usage:{:6.2f} mins.'
                print(msg.format(int(step / 1000), dis_loss, gen_loss, time_diff / 60.))

        self.gen_image(sess)

    def gen_image(self, sess):
        sample_noise = np.random.uniform(-1, 1, size=(25, Config.noise_size))
        samples = sess.run(
            self.generator_graph(self.noise, Config.num_units, Config.size, Config.alpha, reuse=True),
            feed_dict={self.noise: sample_noise})

        plt.figure(figsize=(8, 8), dpi=80)
        for i in range(25):
            img = samples[i]
            plt.subplot(5, 5, i + 1)
            plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
            plt.axis('off')
        plt.show()


if __name__ == "__main__":
    Gan()
