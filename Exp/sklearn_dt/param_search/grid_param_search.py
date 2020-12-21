# -*- coding: utf-8 -*-
# @Time: 2020/12/3 19:53
# @Author: Rollbear
# @Filename: grid_param_search.py
# 网格参数搜索

from tqdm import tqdm  # 进度条支持
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score  # 模型评价
import matplotlib.pyplot as plt
from sklearn import preprocessing
from util.data_interface import *
from util.preprocessing import *
from util.values import DataProperty


def main():
    # ==============================================
    # 数据加载与预处理
    # ==============================================
    scaler = StandardScaler()

    raw_x_train, y_train = split_x_y(drop_cols(get_train()))
    x_train = scaler.fit_transform(raw_x_train)

    raw_x_test, y_test = split_x_y(drop_cols(get_test()))
    x_test = scaler.transform(raw_x_test)

    # ==============================================
    # 参数区间定义
    # ==============================================
    params_max_depth = range(1, 30)  # 树最大深度参数
    params_min_samples_leaf = range(1, 6)  # 节点分裂最小样本数
    params_max_leaf_nodes = range(10, 30)  # 最大叶节点限制

    # ==============================================
    # 网格参数搜索
    # ==============================================
    f1_lt = []
    for max_depth in tqdm(params_max_depth):
        for min_samples_leaf in params_min_samples_leaf:
            for max_leaf_nodes in params_max_leaf_nodes:
                params = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_leaf_nodes": max_leaf_nodes
                }
                clf = DecisionTreeClassifier(**params)
                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)
                f1 = f1_score(y_true=y_test, y_pred=pred)
                f1_lt.append([f1, params])

    best_params = max(f1_lt, key=lambda elem: elem[0])

    print(f"max f1: {best_params[0]}")
    print(f"cur params: {str(best_params[1])}")
    plt.plot([elem[0] for elem in f1_lt])
    plt.show()


if __name__ == '__main__':
    # logging.basicConfig(
    #     filename="../resource/log/param_search.log",
    #     format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    # )
    main()
