# coding=utf-8

import numpy as np
import random
import matplotlib.pyplot as plt

DATA_NUM = 100  # num of data
DATA_DIM = 10  # dimension of data
EXTANT = 10  # x in [-EXTANT, EXTANT]
ITER_NUM = 10000


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
                 max_iter=1e6,
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


def draw_hist(x, name, total_num=ITER_NUM):
    # 频率直方图
    plt.figure()
    plt.hist(x, bins='auto', alpha=0.7, rwidth=0.9, log=False)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(name)
    plt.ylabel("frequency (total = %d)" % total_num)
    plt.title("frequency histogram of " + name)
    plt.savefig(name)
    # plt.show()


def main():

    print('DATA_NUM:', DATA_NUM)
    print('DATA_DIM:', DATA_DIM)
    T = []
    t = []

    for i in range(ITER_NUM):
        print('\riter: %d / %d' % (i + 1, ITER_NUM), end='', flush=True)
        x, y, wf = get_data(DATA_NUM, DATA_DIM, EXTANT)
        x_hat = np.vstack((x, np.ones(DATA_NUM)))
        max_xhat_norm = max([np.linalg.norm(y) for y in np.transpose(x)])
        min_y_wf_xhat = min(y * np.dot(wf, x_hat))
        T.append(max_xhat_norm / min_y_wf_xhat)

        p = Perceptron()
        stop = False
        while not stop:
            stop = p.update(x, y)
        t.append(p.iter_num)

    print()
    print('meant/meanT: %.2f/%.2f' % (np.mean(t),np.mean(T)))

    draw_hist(t, 't')
    draw_hist(T, 'T')
    draw_hist(np.log(t), 'log_t')
    draw_hist(np.log(T), 'log_T')


if __name__ == '__main__':
    main()
