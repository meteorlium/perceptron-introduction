import numpy as np
import matplotlib.pyplot as plt
import random


class Perceptron(object):
    """感知机简单实现

    实现了二维感知机逐步画图，以及高维感知机运行时间统计

    Attributes:

        learning rate: 学习率
        wt: 权重weight
        max iter: 最大迭代次数，用于终止条件
        n iter: 记录当前迭代次数
    """

    def __init__(self,
                 learning_rate=1,
                 data_dim=2,
                 max_iter=1000,
                 n_iter=0):
        self.learning_rate = learning_rate
        self.wt = np.zeros(shape=data_dim + 1)
        self.max_iter = max_iter
        self.n_iter = n_iter

    def update(self, x, y):
        """迭代一次权重wf

        根据数据x，y，随机排序，并查找一个错误分类点。
        若有错误分类点，则修正wt，并返回`stop=false`；
        若没有错误分类点，则返回`stop=true`
        """
        x_num_list = np.arange(x.shape[0])  # rand order
        random.shuffle(x_num_list)
        stop = True
        self.n_iter += 1
        for x_num in x_num_list:  # iterate wt
            if self.n_iter > self.max_iter:
                print('perceptron over max iter')
                break
            wt_dot_x = np.dot(
                self.wt[:-1], x[x_num, :].transpose()) + self.wt[-1]
            if wt_dot_x * y[x_num] <= 0:
                self.wt[:-1] += self.learning_rate * \
                    y[x_num] * x[x_num, :].transpose()
                self.wt[-1] += self.learning_rate * y[x_num]
                stop = False
                break
        return stop

    def train(self, x, y):
        """训练一个感知机

        根据数据x，y，使用update函数迭代，直到达到终止条件（全部分类正确或达到最大迭代次数）时终止，得到最终权重wt，并对其标准化。
        """
        stop = False
        while not stop:
            stop = self.update(x, y)
        self.wt /= sum(self.wt)
