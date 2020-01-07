# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import random

DATA_NUM = 20  # num of data
DATA_DIM = 2  # dimension of data
EXTANT = 10  # x in [-EXTANT, EXTANT]


def get_data(num, dim, ext):
    """
    获得线性可分的随机数据x N个,标记y,分割超平面wf
    num : number of x
    dim : dimension of x
    ext : extent, x in [-extant,extant]
    x : data, dim*num
    y : label of x
    wf : perfect Hyperplane
    """
    wf = np.random.random(size=dim + 1)
    wf /= sum(wf)
    x = np.random.randint(low=-ext, high=ext, size=[dim, num])
    wf_dot_x = np.dot(wf[:-1], x) + wf[-1]

    positive_id = np.where(wf_dot_x > 0)
    negative_id = np.where(wf_dot_x < 0)

    y = np.zeros(num)
    y[positive_id] = 1
    y[negative_id] = -1

    return x, y, wf


class Perceptron:

    def __init__(self,
                 learning_rate=1,
                 wt=np.zeros(shape=DATA_DIM + 1),
                 max_iter=1000,
                 iter_num=0):
        self.learning_rate = learning_rate  # 学习率
        self.wt = wt  # wt,初始化为zeros
        self.max_iter = max_iter  # 最大迭代次数
        self.iter_num = iter_num  # 当前迭代次数

    def update(self, x, y):
        x_num_list = np.arange(DATA_NUM)  # rand order
        random.shuffle(x_num_list)
        stop = True
        self.iter_num += 1
        for x_num in x_num_list:  # iterate wt
            if self.iter_num > self.max_iter:
                print('perceptron over max iter')
                break
            wt_dot_x = np.dot(self.wt[:-1], x[:, x_num]) + self.wt[-1]
            if wt_dot_x * y[x_num] <= 0:
                self.wt[:-1] += self.learning_rate * y[x_num] * x[:, x_num]
                self.wt[-1] += self.learning_rate * y[x_num]
                stop = False
                break
        return stop


def draw_line(w, color, label):
    if not (w == np.zeros(DATA_DIM + 1)).all():
        if w[1] != 0:
            x = [-EXTANT, EXTANT]
            y = [-(-EXTANT * w[0] + w[2]) / w[1], -(EXTANT * w[0] + w[2]) / w[1]]
        else:
            x = [-w[2] / w[0], -w[2] / w[0]]
            y = [-EXTANT, EXTANT]
        plt.plot(x, y, color=color, label=label)


def draw(wf, wt, positive_x, negative_x, iter_num):
    plt.clf()
    draw_line(wf, 'green', 'wf')
    plt.scatter(positive_x[0, :], positive_x[1, :], color='red', label='positive')
    plt.scatter(negative_x[0, :], negative_x[1, :], color='blue', label='negative')
    plt.axis([-EXTANT, EXTANT, -EXTANT, EXTANT])
    draw_line(wt, 'brown', 'wt')
    plt.title('iteration %d' % iter_num)
    plt.legend()
    print('iteration %d!' % iter_num)
    # plt.show()
    save_name = 'iter' + str(iter_num)
    plt.savefig(save_name)


def main():
    x, y, wf = get_data(DATA_NUM, DATA_DIM, EXTANT)
    print('x:', x)
    print('y:', y)
    print('wf:', wf)

    positive_id = np.where(y > 0)
    negative_id = np.where(y < 0)

    print('positive_id:', positive_id)
    print('negative_id:', negative_id)

    p = Perceptron()

    stop = False
    while not stop:
        draw(wf, p.wt, x[:, positive_id], x[:, negative_id], p.iter_num)
        stop = p.update(x, y)
    p.wt /= sum(p.wt)
    print('iter_num:', p.iter_num)
    print('wt:', p.wt)

    wt_dot_x = np.dot(p.wt[:-1], x) + p.wt[-1]
    positive_id = np.where(wt_dot_x > 0)
    negative_id = np.where(wt_dot_x < 0)

    print('positive_id:', positive_id)
    print('negative_id:', negative_id)


if __name__ == '__main__':
    main()
