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

import graphviz
import pydotplus
from sklearn import tree
from IPython.display import Image

from util.preprocessing import *


def main():
    raw_x_train, y_train = split_x_y(drop_cols(get_train()))
    x_train = scale(raw_x_train)

    raw_x_test, y_test = split_x_y(drop_cols(get_test()))
    x_test = scale(raw_x_test)

    dt = DecisionTreeClassifier()
    # dt = RandomForestClassifier()
    # dt = SVC()

    # cross validation, 10 folds, 10 cross
    score_lt = []
    kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    for train_index, test_index in kf.split(x_train):
        dt.fit(x_train[train_index], y_train[train_index])
        pred = dt.predict(x_train[test_index])
        score_lt.append(f1_score(y_train[test_index], pred))

        # heatmap()
        # plt.plot(dt.feature_importances_)
        # plt.show()

    # native f1-score: 0.6974379461520825
    print(sum(score_lt) / len(score_lt))

    # 可视化 via Jieling Lin
    feature_name = [
        "account_length",
        "international_plan",
        "voice_mail_plan",
        "number_vmail_messages",
        "total_day_minutes",
        "total_day_calls",
        "total_day_charge",
        "total_eve_minutes",
        "total_eve_calls",
        "total_eve_charge",
        "total_night_minutes",
        "total_night_calls",
        "total_night_charge",
        "total_intl_minutes",
        "total_intl_calls",
        "total_intl_charge",
        "number_customer_service_calls"
    ]
    target_name = ["0", "1"]
    # 虚拟环境的PATH需要动态加载
    os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin'
    dot_tree = tree.export_graphviz(dt, out_file=None, feature_names=feature_name,
                                    class_names=target_name, filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_tree)
    img = Image(graph.create_png())
    graph.write_png("./resource/tree.png")


if __name__ == '__main__':
    main()
