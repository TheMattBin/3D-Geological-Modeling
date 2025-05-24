import numpy as np
import pandas as pd
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from collections import defaultdict


def acc_com_by_point(dis: float, csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    litoPoints = []
    for _, values in df.iterrows():
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
    soil_class = litoNp[:, 3]
    X_train, X_test, y_train, y_test = train_test_split(coor_trans, soil_class, test_size=0.3, random_state=40)
    clf_knn = neighbors.KNeighborsClassifier(15, weights='distance')
    clf_svm = svm.SVC(kernel='rbf', probability=True, gamma=0.5)
    clf_knn.fit(X_train, y_train)
    clf_svm.fit(X_train, y_train)
    print('Train acc (kNN, SVM):', accuracy_score(y_train, clf_knn.predict(X_train)), accuracy_score(y_train, clf_svm.predict(X_train)))
    print('Test acc (kNN, SVM):', accuracy_score(y_test, clf_knn.predict(X_test)), accuracy_score(y_test, clf_svm.predict(X_test)))
    print('Confusion matrix (train, kNN):\n', confusion_matrix(y_train, clf_knn.predict(X_train)))
    print('Confusion matrix (train, SVM):\n', confusion_matrix(y_train, clf_svm.predict(X_train)))
    print('Confusion matrix (test, kNN):\n', confusion_matrix(y_test, clf_knn.predict(X_test)))
    print('Confusion matrix (test, SVM):\n', confusion_matrix(y_test, clf_svm.predict(X_test)))
    print('Classification report (train, kNN):\n', classification_report(y_train, clf_knn.predict(X_train)))
    print('Classification report (train, SVM):\n', classification_report(y_train, clf_svm.predict(X_train)))
    print('Classification report (test, kNN):\n', classification_report(y_test, clf_knn.predict(X_test)))
    print('Classification report (test, SVM):\n', classification_report(y_test, clf_svm.predict(X_test)))


def acc_com_by_bh(delta: float, csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    litoPoints = defaultdict(list)
    for _, values in df.iterrows():
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
    data_s = pd.Series(litoPoints)
    training_data, test_data = [i.to_dict() for i in train_test_split(data_s, train_size=0.7, random_state=40)]
    X_train, X_test, y_train, y_test = [], [], [], []
    for val in training_data.values():
        for v in val:
            X_train.append(v[:3])
            y_train.append(v[3])
    for val in test_data.values():
        for v in val:
            X_test.append(v[:3])
            y_test.append(v[3])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    clf_knn = neighbors.KNeighborsClassifier(15, weights='distance')
    clf_svm = svm.SVC(kernel='rbf', probability=True, gamma=0.5, decision_function_shape='ovo')
    clf_rf = RandomForestClassifier(45)
    clf_gbc = GradientBoostingClassifier(n_estimators=440)
    clf_knn.fit(X_train, y_train)
    clf_svm.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)
    clf_gbc.fit(X_train, y_train)
    print('Train acc (SVM):', accuracy_score(y_train, clf_svm.predict(X_train)))
    print('Test acc (SVM):', accuracy_score(y_test, clf_svm.predict(X_test)))
    print('Confusion matrix (train, SVM):\n', confusion_matrix(y_train, clf_svm.predict(X_train)))
    print('Confusion matrix (test, SVM):\n', confusion_matrix(y_test, clf_svm.predict(X_test)))
    print('Classification report (train, SVM):\n', classification_report(y_train, clf_svm.predict(X_train)))
    print('Classification report (test, SVM):\n', classification_report(y_test, clf_svm.predict(X_test)))
    print('Kappa (train, SVM):', cohen_kappa_score(y_train, clf_svm.predict(X_train)))
    print('Kappa (test, SVM):', cohen_kappa_score(y_test, clf_svm.predict(X_test)))


def main():
    csv_path = 'CentralWater_20211230_comb.csv'
    for delta in [1, 10, 100, 1000, 10000]:
        print(f"Delta: {delta}")
        acc_com_by_bh(delta, csv_path)


if __name__ == "__main__":
    main()
