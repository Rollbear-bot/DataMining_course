# -*- coding: utf-8 -*-
# @Time: 2020/11/19 19:06
# @Author: Rollbear
# @Filename: sklearn_dt.py
import os

import sklearn
from sklearn.tree import DecisionTreeClassifier  # 基础决策树分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, f1_score  # 模型评价
from sklearn.model_selection import RepeatedKFold  # 交叉验证
from seaborn import heatmap
import matplotlib.pyplot as plt

from util.preprocessing import *
from util.tree_viz import dump_img


def main():
    raw_x_train, y_train = split_x_y(drop_cols(get_train()))
    x_train = scale(raw_x_train)

    # raw_x_test, y_test = split_x_y(drop_cols(get_test()))
    # x_test = scale(raw_x_test)

    # can adopt to other model here.
    chosen_model = DecisionTreeClassifier
    # chosen_model = RandomForestClassifier
    # chosen_model = SVC

    # ==============================================
    # 调参
    # ==============================================
    params_max_depth = range(1, 18)  # 树最大深度参数
    avg_score_lt = []
    for param_max_depth in params_max_depth:
        # cross validation, 10 folds, 10 cross
        score_lt = []
        dt = None
        kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
        for train_index, test_index in kf.split(x_train):
            dt = chosen_model(max_depth=param_max_depth)
            dt.fit(x_train[train_index], y_train[train_index])
            pred = dt.predict(x_train[test_index])
            score_lt.append(f1_score(y_train[test_index], pred))

        # native f1-score: 0.6974379461520825
        cur_iter_f1 = sum(score_lt) / len(score_lt)
        avg_score_lt.append(cur_iter_f1)
        print(f"max_depth: {param_max_depth}\tf1-score: {cur_iter_f1}")

    plt.plot(params_max_depth, avg_score_lt)
    plt.xlabel("max_depth")
    plt.ylabel("f1_score")
    plt.show()

    # 树可视化
    # dump_img(dt, "tree_2.png")


if __name__ == '__main__':
    main()
