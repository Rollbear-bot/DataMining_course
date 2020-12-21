# -*- coding: utf-8 -*-
# @Time: 2020/12/20 9:53
# @Author: Rollbear
# @Filename: bo_param_search.py
# 第三方包
# windows runtime错误

from bayes_optim import BO, ContinuousSpace
from bayes_optim.Surrogate import GaussianProcess
import numpy as np


def demo_2():
    dim = 5
    space = ContinuousSpace([-5, 5]) * dim  # create the search space

    hyperparameters of the GPR model
    thetaL = 1e-10 * (ub - lb) * np.ones(dim)
    thetaU = 10 * (ub - lb) * np.ones(dim)
    model = GaussianProcess(  # create the GPR model
        thetaL=thetaL, thetaU=thetaU
    )

    opt = BO(
        search_space=space,  # 搜索空间
        obj_fun=fitness,  # 目标函数
        model=model,  # 搜索模型
        DoE_size=5,  # number of initial sample points
        max_FEs=50,  # maximal function evaluation
        verbose=True
    )
    opt.run()


def bo():
    pass


if __name__ == '__main__':
    demo_2()
