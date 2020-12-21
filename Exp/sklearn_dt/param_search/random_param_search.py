# -*- coding: utf-8 -*-
# @Time: 2020/12/20 10:46
# @Author: Rollbear
# @Filename: random_param_search.py

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

parameters = {'max_features': ['auto',5, 0.8, None],
              'max_depth': [None, 3, 9]}
# 定义交叉验证机制
cv = StratifiedKFold(n_splits=3, shuffle=True)
# 定义参数搜索器
rf_gridsearch = GridSearchCV(rf_model, parameters, n_jobs=16, cv=cv, scoring='roc_auc',
                                      verbose=2, refit=True)