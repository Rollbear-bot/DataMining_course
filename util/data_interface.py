# -*- coding: utf-8 -*-
# @Time: 2020/11/19 19:06
# @Author: Rollbear
# @Filename: data_interface.py

train_file_path = "../Exp/churn_training.txt"
test_file_path = "../Exp/churn_test.txt"

import pandas as pd


def get_train():
    return pd.read_table(train_file_path)


def get_test():
    return pd.read_table(test_file_path)
