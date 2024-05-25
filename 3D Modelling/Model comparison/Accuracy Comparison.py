import numpy as np
import pandas as pd
from sklearn import neighbors
import netCDF4
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn import svm
import rasterio
from rasterio.plot import show
import numpy.ma as ma
from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def accCombyPt(dis):
    df = pd.read_csv(r'D:\pythonProject\BH Data Processing\CentralWater_20211230_comb.csv')

    # Point cloud of lithologies
    litoPoints = []

    for index, values in df.iterrows():
        wellX, wellY = values.Easting, values.Northing
        wellXY = [wellX, wellY]
        litoPoints.append(wellXY + [values['Ground Level'] - values['Depth Top'], values['Legend Code']])
        litoPoints.append(wellXY + [values['Ground Level'] - values['Depth Base'], values['Legend Code']])

        litoLength = (values['Ground Level'] - values['Depth Top']) - (values['Ground Level'] - values['Depth Base'])
        if litoLength < 1:
            midPoint = wellXY + [(values['Ground Level'] - values['Depth Top']) - litoLength / 2, values['Legend Code']]
        else:
            npoints = int(litoLength)
            for point in range(1, npoints + 1):
                disPoint = wellXY + [
                    (values['Ground Level'] - values['Depth Top']) - litoLength * point / (npoints + 1),
                    values['Legend Code']]
                litoPoints.append(disPoint)

    litoNp = np.array(litoPoints)
    coor_trans = np.hstack((litoNp[:, :2] / dis, litoNp[:, 2].reshape(-1, 1)))
    soil_class = litoNp[:,3]


    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(coor_trans, soil_class, test_size=0.3, random_state=40)

    # define lists to collect scores
    train_scores, test_scores = list(), list()


    # we create an instance of Neighbours Classifier and fit the data.
    clf_knn = neighbors.KNeighborsClassifier(15, weights='distance')
    clf_svm = svm.SVC(kernel='rbf', probability=True, gamma=0.5)
    clf_knn.fit(X_train, y_train)
    clf_svm.fit(X_train, y_train)
    train_yhat_knn = clf_knn.predict(X_train)
    train_acc_knn = accuracy_score(y_train, train_yhat_knn)
    train_yhat_svm = clf_svm.predict(X_train)
    train_acc_svm = accuracy_score(y_train, train_yhat_svm)
    print(train_acc_knn, train_acc_svm)

    # evaluate on the test dataset
    test_yhat_knn = clf_knn.predict(X_test)
    test_acc_knn = accuracy_score(y_test, test_yhat_knn)
    test_yhat_svm = clf_svm.predict(X_test)
    test_acc_svm = accuracy_score(y_test, test_yhat_svm)
    print(test_acc_knn, test_acc_svm)


    result_knn = confusion_matrix(y_train, train_yhat_knn)
    result_svm = confusion_matrix(y_train, train_yhat_svm)
    print(result_knn, result_svm)

    result_knn_test = confusion_matrix(y_test, test_yhat_knn)
    result_svm_test = confusion_matrix(y_test, test_yhat_svm)
    print(result_knn_test, result_svm_test)

    report_knn_train = classification_report(y_train, train_yhat_knn)
    report_svm_train = classification_report(y_train, train_yhat_svm)
    print(report_knn_train, report_svm_train)

    report_knn_test = classification_report(y_test, test_yhat_knn)
    report_svm_test = classification_report(y_test, test_yhat_svm)
    print(report_knn_test, report_svm_test)

def AccCombyBH(delta):
    df = pd.read_csv(r'D:\pythonProject\BH Data Processing\CentralWater_20211230_comb.csv')

    # Point cloud of lithologies
    litoPoints = defaultdict(list)

    for index, values in df.iterrows():
        wellX, wellY = values.Easting, values.Northing
        wellXY = [wellX / delta, wellY / delta]
        litoPoints[values['Location ID']].append(
            wellXY + [values['Ground Level'] - values['Depth Top'], values['Legend Code']])
        litoPoints[values['Location ID']].append(
            wellXY + [values['Ground Level'] - values['Depth Base'], values['Legend Code']])

        litoLength = (values['Ground Level'] - values['Depth Top']) - (values['Ground Level'] - values['Depth Base'])
        if litoLength < 1:
            midPoint = wellXY + [(values['Ground Level'] - values['Depth Top']) - litoLength / 2, values['Legend Code']]
            litoPoints[values['Location ID']].append(midPoint)
        else:
            npoints = int(litoLength)
            for point in range(1, npoints + 1):
                disPoint = wellXY + [
                    (values['Ground Level'] - values['Depth Top']) - litoLength * point / (npoints + 1),
                    values['Legend Code']]
                litoPoints[values['Location ID']].append(disPoint)
    print(len(litoPoints))
    data_s = pd.Series(litoPoints)
    training_data, test_data = [i.to_dict() for i in train_test_split(data_s, train_size=0.7, random_state=40)]
    print(len(training_data), len(test_data))

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for key, val in training_data.items():
        for v in val:
            X_train.append(v[:3])
            y_train.append(v[3])
    for key, val in test_data.items():
        for v in val:
            X_test.append(v[:3])
            y_test.append(v[3])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    clf_knn = neighbors.KNeighborsClassifier(15, weights='distance')
    clf_svm = svm.SVC(kernel='rbf', probability=True, gamma=0.5, decision_function_shape='ovo')
    clf_rf = RandomForestClassifier(45)
    clf_gbc = GradientBoostingClassifier(n_estimators=440)

    clf_knn.fit(X_train, y_train)
    clf_svm.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)
    clf_gbc.fit(X_train, y_train)

    train_yhat_knn = clf_knn.predict(X_train)
    train_yhat_svm = clf_svm.predict(X_train)
    train_yhat_rf = clf_rf.predict(X_train)
    train_yhat_gbc = clf_gbc.predict(X_train)
    train_acc_knn = accuracy_score(y_train, train_yhat_knn)
    train_acc_svm = accuracy_score(y_train, train_yhat_svm)
    train_acc_rf = accuracy_score(y_train, train_yhat_rf)
    train_acc_gbc = accuracy_score(y_train, train_yhat_gbc)

    confusion_tr_knn = confusion_matrix(y_train, train_yhat_knn)
    confusion_tr_svm = confusion_matrix(y_train, train_yhat_svm)
    confusion_tr_rf = confusion_matrix(y_train, train_yhat_rf)
    confusion_tr_gbc = confusion_matrix(y_train, train_yhat_gbc)

    conf_rep_train_knn = classification_report(y_train, train_yhat_knn)
    conf_rep_train_svm = classification_report(y_train, train_yhat_svm)
    conf_rep_train_rf = classification_report(y_train, train_yhat_rf)
    conf_rep_train_gbc = classification_report(y_train, train_yhat_gbc)

    kappa_val_train_knn = cohen_kappa_score(y_train, train_yhat_knn)
    kappa_val_train_svm = cohen_kappa_score(y_train, train_yhat_svm)
    # print(train_acc_knn, train_acc_svm)
    # print(confusion_tr_knn, confusion_tr_svm)
    # print(conf_rep_train_knn, conf_rep_train_svm)
    # print(kappa_val_train_knn, kappa_val_train_svm)
    # print(conf_rep_train_gbc)

    # evaluate on the test dataset
    test_yhat_knn = clf_knn.predict(X_test)
    test_yhat_svm = clf_svm.predict(X_test)
    test_yhat_rf = clf_rf.predict(X_test)
    test_yhat_gbc = clf_rf.predict(X_test)

    test_acc_knn = accuracy_score(y_test, test_yhat_knn)
    test_acc_svm = accuracy_score(y_test, test_yhat_svm)

    confusion_t_knn = confusion_matrix(y_test, test_yhat_knn)
    confusion_t_svm = confusion_matrix(y_test, test_yhat_svm)

    conf_rep_test_knn = classification_report(y_test, test_yhat_knn)
    conf_rep_test_svm = classification_report(y_test, test_yhat_svm)
    conf_rep_test_rf = classification_report(y_test, test_yhat_rf)
    conf_rep_test_gbc = classification_report(y_test, test_yhat_gbc)

    kappa_val_test_knn = cohen_kappa_score(y_test, test_yhat_knn)
    kappa_val_test_svm = cohen_kappa_score(y_test, test_yhat_svm)
    # print(test_acc_knn, test_acc_svm)
    # print(confusion_t_knn, confusion_t_svm)
    # print(conf_rep_test_knn, conf_rep_test_svm)
    # print(kappa_val_test_knn, kappa_val_test_svm)
    # print(conf_rep_test_rf)
    # print(conf_rep_test_gbc)
    print(train_acc_svm, test_acc_svm, confusion_tr_svm, confusion_t_svm)



for i in [1, 10, 100, 1000, 10000]:
    print(i)
    AccCombyBH(i)
