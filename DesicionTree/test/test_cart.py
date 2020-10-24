# -*- coding: utf-8 -*-
# @Time: 2020/10/24 10:42
# @Author: Rollbear
# @Filename: test_cart.py

from unittest import TestCase
import unittest

from ..CART_DT import *


class TestCART(TestCase):
    def test_info_gain(self):
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
        y = [row[-1] for row in dataset]

        res = gini_info_gain(feature_index=0, x=x, y=y)
        assert res == {"男": 10, "女": 10}


if __name__ == '__main__':
    unittest.main()
