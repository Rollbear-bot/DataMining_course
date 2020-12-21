# -*- coding: utf-8 -*-
# @Time: 2020/11/19 19:06
# @Author: Rollbear
# @Filename: data_interface.py

import pandas as pd

from util.values import DataProperty


def get_train():
    return pd.read_table(DataProperty.train_file_path)


def get_test():
    return pd.read_table(DataProperty.test_file_path)
