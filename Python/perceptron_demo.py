import numpy as np
import random

DATA_NUM = 20  # num of data
DATA_DIM = 2  # dimension of data
EXTANT = 10  # x in [-EXTANT, EXTANT]


def get_data(num, dim, ext):
    """获得线性可分的随机数据x(num个,dim维数,[-ext,ext]范围),标记y,分割超平面wf"""
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


def main():
    x, y, wf = get_data(DATA_NUM, DATA_DIM, EXTANT)
    p = Perceptron()
    # perceptron iteration
    stop = False
    while not stop:
        stop = p.update(x, y)
    # print solution wt
    p.wt /= sum(p.wt)
    print(p.iter_num)
    print(p.wt)


if __name__ == '__main__':
    main()
