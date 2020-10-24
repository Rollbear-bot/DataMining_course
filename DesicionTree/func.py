# -*- coding: utf-8 -*-
# @Time: 2020/10/22 21:07
# @Author: Rollbear
# @Filename: func.py


def gini(p_arr):
    """
    计算基尼指数
    :param p_arr: 对每个类别，在样本t处发生的频次
    :return: 基尼指数
    """
    return 1 - sum([elem ** 2 for elem in p_arr])
