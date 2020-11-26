# -*- coding: utf-8 -*-
# @Time: 2020/11/19 19:27
# @Author: Rollbear
# @Filename: preprocessing.py

from sklearn import preprocessing
import pandas as pd

from util.data_interface import *


def scale(data: pd.DataFrame):
    return preprocessing.scale(data)


def drop_cols(data: pd.DataFrame):
    """去除state, area_code, and phone_number三列"""
    return data.drop(data.columns[[0, 2, 3]], axis=1)


def split_x_y(data: pd.DataFrame):
    """分离"""
    y_col_id = -1
    y = data.values[:, y_col_id]
    x = data.drop(data.columns[[y_col_id]], axis=1)
    return x, y
