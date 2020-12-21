# -*- coding: utf-8 -*-
# @Time: 2020/11/26 15:46
# @Author: Rollbear
# @Filename: sklearn_dt_predict.py

from util.data_interface import *
from util.preprocessing import *
from util.tree_viz import dump_img
from util.values import DataProperty

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score  # 模型评价
from sklearn.metrics import roc_curve, auc


def main():
    scaler = StandardScaler()

    raw_x_train, y_train = split_x_y(drop_cols(get_train()))
    x_train = scaler.fit_transform(raw_x_train)

    raw_x_test, y_test = split_x_y(drop_cols(get_test()))
    x_test = scaler.transform(raw_x_test)

    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(x_train, y_train)

    pred = dt.predict(x_test)

    print("clf report:")
    print(classification_report(y_test, pred))

    print("auc:")
    fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=2)
    print(auc(fpr, tpr))

    print("f1 on test set: ", f1_score(y_test, pred))
    print("f1 on train set: ", f1_score(y_train, dt.predict(x_train)))

    # dump_img(dt, "tree_predict.png")


if __name__ == '__main__':
    main()

