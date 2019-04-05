# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2019-04-05
# @version: python3.7

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime


class Config:
    batch_size = 64
    epoch = 10
    momentum = 0.9
    alpha = 1e-3

    print_per_step = 100


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # 加快收敛速度的方法（注：批标准化一般放在全连接层后面，激活函数层的前面）
            nn.ReLU()
        )

        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class TrainProcess:

    def __init__(self):
        self.train, self.test = self.load_data()
        self.net = LeNet()
        self.criterion = nn.CrossEntropyLoss()  # 定义损失函数
        self.optimizer = optim.SGD(self.net.parameters(), lr=Config.alpha, momentum=Config.momentum)

    @staticmethod
    def load_data():
        print("Loading Data......")
        """加载MNIST数据集，本地数据不存在会自动下载"""
        train_data = datasets.MNIST(root='./data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

        test_data = datasets.MNIST(root='./data/',
                                   train=False,
                                   transform=transforms.ToTensor())

        # 返回一个数据迭代器
        # shuffle：是否打乱顺序
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=Config.batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=Config.batch_size,
                                                  shuffle=False)
        return train_loader, test_loader

    def train_step(self):
        steps = 0
        start_time = datetime.now()

        print("Training & Evaluating......")
        for epoch in range(Config.epoch):
            print("Epoch {:3}".format(epoch + 1))

            for data, label in self.train:
                data, label = Variable(data.cpu()), Variable(label.cpu())
                self.optimizer.zero_grad()  # 将梯度归零
                outputs = self.net(data)  # 将数据传入网络进行前向运算
                loss = self.criterion(outputs, label)  # 得到损失函数
                loss.backward()  # 反向传播
                self.optimizer.step()  # 通过梯度做一步参数更新

                # 每100次打印一次结果
                if steps % Config.print_per_step == 0:
                    _, predicted = torch.max(outputs, 1)
                    correct = int(sum(predicted == label))
                    accuracy = correct / Config.batch_size  # 计算准确率
                    end_time = datetime.now()
                    time_diff = (end_time - start_time).seconds
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                    print(msg.format(steps, loss, accuracy, time_usage))

                steps += 1

        test_loss = 0.
        test_correct = 0
        for data, label in self.test:
            data, label = Variable(data.cpu()), Variable(label.cpu())
            outputs = self.net(data)
            loss = self.criterion(outputs, label)
            test_loss += loss * Config.batch_size
            _, predicted = torch.max(outputs, 1)
            correct = int(sum(predicted == label))
            test_correct += correct

        accuracy = test_correct / len(self.test.dataset)
        loss = test_loss / len(self.test.dataset)
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))


if __name__ == "__main__":
    p = TrainProcess()
    p.train_step()
