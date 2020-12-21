# -*- coding: utf-8 -*-
# @Time: 2020/11/19 19:27
# @Author: Rollbear
# @Filename: preprocessing.py

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE

from util.data_interface import *


# todo::其中一个类标签样本很少，是否需要数据增广？


def drop_cols(data: pd.DataFrame):
    """去除state, area_code, and phone_number三列"""
    return data.drop(data.columns[[0, 2, 3]], axis=1)


def split_x_y(data: pd.DataFrame):
    """分离xy"""
    y_col_id = -1
    y = data.values[:, y_col_id]
    x = data.drop(data.columns[[y_col_id]], axis=1)
    return x, y


# todo::用PCA做特征筛选
def d_reduce(data: pd.DataFrame):
    """数据降维"""
    # PCA模型参数
    # n_components = 2
    pca = PCA()
    pca.fit_transform(data)
    inv_x = pca.inverse_transform(data)
    return inv_x


def boderline_SMOTE(x_train, y_train):
    print('Original dataset shape %s' % Counter(y_train))
    ada = BorderlineSMOTE(random_state=42, kind="borderline-1")
    x_res, y_res = ada.fit_resample(x_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_res))
    return x_res, y_res
