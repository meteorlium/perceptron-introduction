# coding=utf-8

import os

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron
from utility import get_data


def draw_hist(x, name, total_num, filename):
    """画频率直方图

    测试感知机对于total num个数据的迭代次数

    Args:

        x (list): data for draw hist
        name (str): for xlabel and title
        total num (int): for ylabel
        filename (str): for save
    """
    plt.figure()
    plt.hist(x, bins='auto', alpha=0.7, rwidth=0.9, log=False)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(name)
    plt.ylabel(f"frequency (total = {total_num:d})")
    plt.title("frequency histogram of " + name)
    path = 'draw_hist'
    plt.savefig(os.path.join(path, filename))
    plt.close()


def calculate_T(x, y, wf):
    """计算理论最大迭代次数T

    T = max( norm([x,1]) ) / min( y * (wf^T * [x,1]) )
    """
    one = np.ones([x.shape[0], 1])
    x_hat = np.hstack((x, one))
    max_xhat_norm = max([np.linalg.norm(y) for y in x.transpose()])
    min_y_wf_xhat = min(y * np.dot(wf, x_hat.transpose()))
    T = max_xhat_norm / min_y_wf_xhat
    return T


def main():
    """高维数据的迭代次数统计，及与理论迭代次数上限的比较
    """
    DATA_NUM = 100
    DATA_DIM = 10
    ITER_NUM = 10000

    print('DATA_NUM:', DATA_NUM)
    print('DATA_DIM:', DATA_DIM)
    T = []
    t = []

    for i in range(ITER_NUM):
        print(f'\riter: {i+1:d} / {ITER_NUM:d}', end='', flush=True)
        x, y, wf = get_data(DATA_NUM, DATA_DIM)
        T.append(calculate_T(x, y, wf))

        p = Perceptron(data_dim=DATA_DIM)
        p.train(x, y)
        t.append(p.n_iter)

    print()
    print(f'meant/meanT: {np.mean(t):.2f}/{np.mean(T):.2f}')

    draw_hist(t, 't', ITER_NUM, 'fig1_t')
    draw_hist(T, 'T', ITER_NUM, 'fig2_T')
    draw_hist(np.log(t), 'log_t', ITER_NUM, 'fig3_log_t')
    draw_hist(np.log(T), 'log_T', ITER_NUM, 'fig4_log_T')


if __name__ == '__main__':
    main()
