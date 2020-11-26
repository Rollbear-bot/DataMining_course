# -*- coding: utf-8 -*-
# @Time: 2020/11/26 8:22
# @Author: Rollbear
# @Filename: homemake_dt.py

from DesicionTree.CART_DT import CART
from util.data_interface import *
from util.preprocessing import *


def main():
    raw_x_train, y_train = split_x_y(drop_cols(get_train()))
    x_train = scale(raw_x_train)

    raw_x_test, y_test = split_x_y(drop_cols(get_test()))
    x_test = scale(raw_x_test)

    dt = CART(max_tree_depth=float("inf"))
    dt.fit(x_train, y_train, debug=True)


if __name__ == '__main__':
    main()
