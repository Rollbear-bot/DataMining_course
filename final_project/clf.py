# -*- coding: utf-8 -*-
# @Time: 2020/12/17 19:26
# @Author: Rollbear
# @Filename: clf.py

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from feature_select import get_data_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

import matplotlib.pyplot as plt


def main():
    n_iter = 1
    n_top_range = range(2, 81)

    f1_lt = []
    for n in tqdm(n_top_range):
        avg_f1 = 0
        for i in range(n_iter):
            x_train, x_test, y_train, y_test \
                = get_data_split(clf_mode=True, feature_select=True, n_top=n)

            clf = RandomForestClassifier()

            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            avg_f1 += f1_score(y_test, pred)
            # print(classification_report(y_test, pred))

        avg_f1 /= n_iter
        print("avg_f1:", avg_f1)
        f1_lt.append(avg_f1)

    plt.plot(f1_lt)
    plt.show()


if __name__ == '__main__':
    main()
