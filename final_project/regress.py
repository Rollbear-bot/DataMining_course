# -*- coding: utf-8 -*-
# @Time: 2020/12/17 19:26
# @Author: Rollbear
# @Filename: regress.py

from feature_select import get_data_split
from sklearn.

def main():
    x_train, x_test, y_train, y_test \
        = get_data_split()

    clf = SVC()

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    # 16, 29
    print(mean_squared_error(y_test, pred))