# -*- coding: utf-8 -*-
# @Time: 2020/11/26 11:12
# @Author: Rollbear
# @Filename: tree_viz.py

import os
import graphviz
import pydotplus
from sklearn import tree
from IPython.display import Image

from util.values import *


def dump_img(sklearn_tree_model, img_name, dir=ResourceRoot.exp_resource_root):
    # 虚拟环境的PATH需要动态加载
    os.environ["PATH"] += os.pathsep + ResourceRoot.graphviz_home
    dot_tree = tree.export_graphviz(sklearn_tree_model, out_file=None, feature_names=DataProperty.feature_name,
                                    class_names=DataProperty.target_name, filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_tree)
    img = Image(graph.create_png())
    graph.write_png(dir + img_name)
