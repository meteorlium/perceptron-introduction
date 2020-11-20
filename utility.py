import numpy as np
import random


def get_data(num, dim):
    """获得数据
    获得线性可分的随机数据x N个,标记y,分割超平面wf

    Args:

        num (int): number of x
        dim (int): dimension of x

    Returns:

        x (np.array): data in (-1, 1)
        y (np.array): label of x
        wf (np.array): perfect Hyperplane
    """
    wf = np.random.random(size=dim + 1)
    wf /= sum(wf)
    x = np.random.rand(num, dim) * 2 - 1
    wf_dot_x = np.dot(wf[:-1], x.transpose()) + wf[-1]

    positive_id = np.where(wf_dot_x > 0)
    negative_id = np.where(wf_dot_x < 0)

    y = np.zeros(num)
    y[positive_id] = 1
    y[negative_id] = -1

    return x, y, wf
