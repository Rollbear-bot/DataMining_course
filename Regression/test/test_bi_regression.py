# -*- coding: utf-8 -*-
# @Time: 2020/10/24 21:18
# @Author: Rollbear
# @Filename: test_bi_regression.py

from sympy import *

x = symbols('x')
solve(x + 2 * (x ** 2) + 3 * (x ** 3) - 6, x)
