import random

import numpy as np

from perceptron import Perceptron
from utility import get_data


def main():
    DATA_NUM = 20
    DATA_DIM = 2
    x, y, wf = get_data(DATA_NUM, DATA_DIM)
    p = Perceptron(data_dim=DATA_DIM)
    p.train(x, y)
    print(p.n_iter)
    print(p.wt)


if __name__ == '__main__':
    main()
