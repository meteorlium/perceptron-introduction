# coding=utf-8
import os

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron
from utility import get_data


def draw_line(w, color, label):
    """画线，draw的子函数
    """
    if not (w == np.zeros(w.shape)).all():
        if w[1] != 0:
            x = [-1, 1]
            y = [-(-1 * w[0] + w[2]) / w[1], -
                 (1 * w[0] + w[2]) / w[1]]
        else:
            x = [-w[2] / w[0], -w[2] / w[0]]
            y = [-1, 1]
        plt.plot(x, y, color=color, label=label)


def draw(wf, wt, positive_x, negative_x, iter_num):
    """画出感知机迭代过程图

    Args:

        wf (np.array): true weight
        wt (np.array): current weight
        positive x (np.array): x with positive label
        negative x (np.array): x with negative label
        iter num (int): current iter num

    Returns:

        nothing but save pictures in save path
    """
    plt.clf()
    draw_line(wf, 'green', 'wf')
    plt.scatter(positive_x[:, 0], positive_x[:, 1],
                color='red', label='positive')
    plt.scatter(negative_x[:, 0], negative_x[:, 1],
                color='blue', label='negative')
    plt.axis([-1, 1, -1, 1])
    draw_line(wt, 'brown', 'wt')
    plt.title(f'iteration {iter_num:d}')
    plt.legend()
    print(f'iteration {iter_num:d}!')
    # plt.show()
    save_path = 'draw'
    save_name = 'iter' + str(iter_num)
    plt.savefig(os.path.join(save_path, save_name))


def main():
    """二维数据的迭代过程作图
    """
    DATA_NUM = 20  # num of data
    DATA_DIM = 2  # dimension of data
    x, y, wf = get_data(DATA_NUM, DATA_DIM)
    print('x:', x)
    print('y:', y)
    print('wf:', wf)

    positive_id = np.where(y > 0)
    negative_id = np.where(y < 0)

    print('true result:')
    print('positive_id:', positive_id)
    print('negative_id:', negative_id)

    p = Perceptron(data_dim=DATA_DIM)

    stop = False
    while not stop:
        draw(wf, p.wt, x[positive_id],
             x[negative_id], p.n_iter)
        stop = p.update(x, y)
    p.wt /= sum(p.wt)

    print('iter num:', p.n_iter)
    print('wt:', p.wt)

    wt_dot_x = np.dot(p.wt[:-1], x.transpose()) + p.wt[-1]
    positive_id = np.where(wt_dot_x > 0)
    negative_id = np.where(wt_dot_x < 0)

    print('perceptron result:')
    print('positive_id:', positive_id)
    print('negative_id:', negative_id)


if __name__ == '__main__':
    main()
