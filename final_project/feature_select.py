# -*- coding: utf-8 -*-
# @Time: 2020/12/17 19:08
# @Author: Rollbear
# @Filename: feature_select.py

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_raw():
    data_path = "./data/final_depression_data.xlsx"
    data = pd.read_excel(data_path)
    y = data.iloc[:, 4]
    x = data.iloc[:, 5:-1]

    return train_test_split(x, y, test_size=0.3)


def get_top_n_f(num_top):
    x_train, x_test, y_train, y_test \
        = get_raw()

    # 迭代10次取重要性前num_top个特征
    f_importance_sum = 0
    for _ in range(10):
        rf = RandomForestRegressor()

        rf.fit(x_train, y_train)
        f_importance_sum += np.array(rf.feature_importances_)

    res = sorted(enumerate(f_importance_sum),
                 key=lambda elem: elem[1],
                 reverse=True)[:num_top]

    return res


def get_data_split(clf_mode=False, feature_select=False, n_top=5):
    n_import_f = get_top_n_f(n_top)
    data_path = "./data/final_depression_data.xlsx"
    data = pd.read_excel(data_path)
    y = data.iloc[:, 4]
    if clf_mode:
        y = [0 if score <= 17 else 1 for score in y]

    # 在此更改特征选取
    if feature_select:
        x = data.iloc[:, [5+f[0] for f in n_import_f]]
    else:
        x = data.iloc[:, 5:-1]

    return train_test_split(x, y, test_size=0.3)