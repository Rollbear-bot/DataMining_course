# -*- coding: utf-8 -*-
# @Time: 2020/10/22 21:19
# @Author: Rollbear
# @Filename: dt_demo.py

from DesicionTree.CART_DT import CART
from sklearn.model_selection import train_test_split


def main():
    dataset = [
        (1, "男", "家用", "小", "C0"),
        (2, "男", "运动", "中", "C0"),
        (3, "男", "运动", "中", "C0"),
        (4, "男", "运动", "大", "C0"),
        (5, "男", "运动", "加大", "C0"),
        (6, "男", "运动", "加大", "C0"),
        (7, "女", "运动", "小", "C0"),
        (8, "女", "运动", "小", "C0"),
        (9, "女", "运动", "中", "C0"),
        (10, "女", "豪华", "大", "C0"),
        (11, "男", "家用", "大", "C1"),
        (12, "男", "家用", "加大", "C1"),
        (13, "男", "家用", "中", "C1"),
        (14, "男", "豪华", "加大", "C1"),
        (15, "女", "豪华", "小", "C1"),
        (16, "女", "豪华", "小", "C1"),
        (17, "女", "豪华", "中", "C1"),
        (18, "女", "豪华", "中", "C1"),
        (19, "女", "豪华", "中", "C1"),
        (20, "女", "豪华", "大", "C1")
    ]

    x = [row[1:-1] for row in dataset]
    x_with_id = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    cart_tree = CART(max_tree_depth=2)
    cart_tree.fit(x_with_id, y, debug=True)

    print(cart_tree.tree_root)
    predicted = cart_tree.predict(x)


if __name__ == '__main__':
    main()
