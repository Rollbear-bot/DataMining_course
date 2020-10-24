# -*- coding: utf-8 -*-
# @Time: 2020/10/22 21:18
# @Author: Rollbear
# @Filename: CART_DT.py

from .func import gini


def gini_info_gain(feature_index, x, y, debug=False):
    """
    计算信息增益（基于基尼指数）
    :param feature_index: 指定的特征的序号
    :param x: 自变量
    :param y: 预测量
    :param debug: 是否显示计算过程
    :return: 信息增益
    """
    col = [item[feature_index] for item in x]  # 取列
    desc_dict = {key: {} for key in set(col)}

    # 针对不同的分支计数
    for row_index, row in enumerate(x):
        desc_dict[row[feature_index]][y[row_index]] = desc_dict[row[feature_index]].get(y[row_index], 0) + 1

    if debug:
        print("-" * 10 + f"Feature index: {feature_index}" + "-" * 10)
        print(f"desc_dict: {desc_dict}")

    # 计算分裂前的基尼系数
    y_count = {}
    for elem in y:
        y_count[elem] = y_count.get(elem, 0) + 1
    class_count = [value / len(y) for value in y_count.values()]
    gini_before = gini(class_count)

    if debug:
        print(f"impurity(gini) before split: {gini_before}")

    # 分裂后的基尼指数：各个分支gini的平均
    branch_gini = [gini([value / sum(d.values()) for value in d.values()]) for d in desc_dict.values()]
    gini_after = sum(branch_gini) / len(desc_dict)

    if debug:
        print(f"impurity(gini) after split: {gini_after}")

    # 信息增益即为分裂前的基尼指数减去分裂后的基尼指数
    return gini_before - gini_after


class Splitter:
    def __init__(self, feature_index):
        self.split_rule = None
        self.feature_index = feature_index

    def split(self, x):
        return self.split_rule(x)


class CART:
    """CART决策树"""
    def __init__(self, max_tree_depth):
        self.max_tree_depth = max_tree_depth
        self.tree_root = None

    def fit(self, x, y, debug=False):
        """
        拟合模型
        :param x: 自变量
        :param y: 预测量
        :param debug: 是否显示计算过程
        :return: None
        """
        self.tree_root = self.make_tree(x, y, debug)

    def make_tree(self, x, y, debug=False, cur_tree_depth=1):
        """
        递归展开决策树
        :param x: 自变量
        :param y: 预测量
        :param debug: 是否显示计算过程
        :param cur_tree_depth: 当前树深度（从深度1开始递归展开树）
        :return: 树根
        """
        # 递归深度不超过模型设置的最大树深
        if cur_tree_depth > self.max_tree_depth:
            return
        if debug:
            print("\n" + "="*10 + f"cur tree depth: {cur_tree_depth}" + "="*10)

        root = []  # 子树的根
        x_buffer = [[elem for elem in item] for item in x]
        # 选取使得信息增益最大的特征
        f_chosen_index = max(range(len(x_buffer[0])),
                             key=lambda f_index: gini_info_gain(f_index, x_buffer, y, debug))

        if debug:
            print(f"feature chosen: {f_chosen_index}")

        root.append(Splitter(f_chosen_index))
        children = {key: [] for key in set([elem[f_chosen_index] for elem in x_buffer])}
        for row_index, row in enumerate(x_buffer):
            children[row[f_chosen_index]].append([row, y[row_index]])
            del row[f_chosen_index]  # 删去已选择的feature

        for branch in children.values():
            # 前序伸展生成子节点
            root.append(self.make_tree(x=[item[0] for item in branch],
                                       y=[item[1] for item in branch],
                                       debug=debug,
                                       cur_tree_depth=cur_tree_depth + 1))
        return root

    def predict(self, x):
        # todo::设置分裂判定规则
        pass
