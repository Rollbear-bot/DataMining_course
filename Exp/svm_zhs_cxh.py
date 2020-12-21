from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

SCORE_LT = []


def scale(data_train: pd.DataFrame, data_test: pd.DataFrame):
    """standardize"""
    scaler = StandardScaler().fit(data_train)
    train = scaler.fit_transform(data_train)
    test = scaler.fit_transform(data_test)
    return train, test


def drop_cols(data: pd.DataFrame):
    """去除state, area_code, and phone_number三列"""
    return data.drop(data.columns[[0, 2, 3]], axis=1)


def split_x_y(data: pd.DataFrame):
    """分离xy"""
    y_col_id = -1
    y = data.values[:, y_col_id]
    x = data.drop(data.columns[[y_col_id]], axis=1)
    return x, y


def get_train(t):
    p = "./churn_training.txt"
    data = pd.read_csv(p, sep="	")
    return data


def pca(x_train: list, x_test: list, i):
    pca = PCA(n_components=i)
    pca.fit(x_train)
    pca.fit(x_test)
    return pca.transform(x_train), pca.transform(x_test)


# SVC参数解释
# （1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
# （2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";
# （3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
# （4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;
# （5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
# （6）probablity: 可能性估计是否使用(true or false)；
# （7）shrinking：是否进行启发式；
# （8）tol（default = 1e - 3）: svm结束标准的精度;
# （9）cache_size: 制定训练所需要的内存（以MB为单位）；
# （10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应；
# （11）verbose: 跟多线程有关，不大明白啥意思具体；
# （12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
# （13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
# （14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。

def pca_svm(x_train, y_train, x_test, if_pca):
    # tunned_parameters = [{'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[1, 10, 100, 1000]}]
    # # scores = ['precision', 'recall']
    # scores = ['precision']
    # for p in tunned_parameters:
    #     print(p)
    #     for score in scores:
    #         clf = GridSearchCV(SVC(), p, cv=5,scoring='%s_macro' % score)

    c_range = range(1, 100)

    for c in tqdm(c_range):
        clf = SVC(C=c, kernel='rbf', cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=3, gamma='auto',
                  max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001, verbose=False)
        clf.fit(x_train, y_train)
        y_true, y_pred = y_test, clf.predict(x_test)
        # method=if_pca+'kernel:'+p['kernel'][0]
        method = if_pca + ' RBF:'
        report_scores(y_test, y_pred, method)


def report_scores(y_test, y_pred, method):
    # 分类错误计数
    # accuracyCount = 0.0
    tpCount = 0
    fpCount = 0
    tnCount = 0
    fnCount = 0

    for i in range(len(y_test)):
        # 混淆矩阵计数
        if y_test[i] == 1 and y_pred[i] == 1:
            tpCount = tpCount + 1

        if y_test[i] == 1 and y_pred[i] == 0:
            fpCount = fpCount + 1

        if y_test[i] == 0 and y_pred[i] == 0:
            tnCount = tnCount + 1

        if y_test[i] == 0 and y_pred[i] == 1:
            fnCount = fnCount + 1
    # print(fnCount)

    # 性能评估计算

    accuracy = (tpCount + tnCount) / float(len(y_test))
    s = "============" + method + "========================"
    print(s)
    print("avg_accuracy:", accuracy)
    precision = -1
    recall = -1
    FPR = -1
    G_Mean = -1
    f_score = -1
    flag = 1

    if (tpCount + fpCount) != 0:
        precision = tpCount / float(tpCount + fpCount)
        print("avg_precision:", precision)
    else:
        flag = 0
    # print("avg_precision:", precision)
    if (tpCount + fnCount) != 0:
        recall = tpCount / float(tpCount + fnCount)
        print("avg_recall:", recall)
    else:
        flag = 0

    if (fpCount + tnCount) != 0:
        FPR = fpCount / float(fpCount + tnCount)
        print("avg_FPR:", FPR)
    else:
        flag = 0

    if flag:
        G_Mean = (recall * (1 - FPR)) ** 0.5
        print("avg_G-Mean:", G_Mean)
    if flag:
        f_score = 2 * ((precision * recall) / (precision + recall))
        print("avg_f_score:", f_score)

        # only for test
        SCORE_LT.append(f_score)

    auc = roc_auc_score(y_true=y_test, y_score=y_pred, average='macro')
    print("avg_AUC:", auc)
    print("=" * len(s))
    return


# def adasyn(x_train,y_train):
#     from collections import Counter
#     from imblearn.over_sampling import BorderlineSMOTE
#
#     print('Original dataset shape %s' % Counter(y_train))
#     ada =BorderlineSMOTE(random_state=42,kind="borderline-1")
#     x_res, y_res = ada.fit_resample(x_train, y_train)
#     print('Resampled dataset shape %s' % Counter(y_res))
#     return x_res,y_res

if __name__ == '__main__':
    raw_x_train, y_train = split_x_y(drop_cols(get_train("churn_training")))
    raw_x_test, y_test = split_x_y(drop_cols(get_train("churn_test")))
    x_train, x_test = scale(raw_x_train, raw_x_test)
    # x_train,y_train=adasyn(x_train, y_train)
    pca_svm(x_train, y_train, x_test, "")

    plt.plot(SCORE_LT)
    plt.show()

    # x_Train,x_Test=pca(x_train,x_test,10)
    # y_pred=pca_svm(x_Train,y_train,x_Test,"PCA")
