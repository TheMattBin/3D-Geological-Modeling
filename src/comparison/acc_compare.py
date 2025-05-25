import numpy as np
import pandas as pd
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
from typing import List, Dict


def acc_com_by_bh(delta: float, csv_path: str, models: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compare accuracy of different models for borehole data and return metrics.
    Args:
        delta (float): Scaling factor for coordinates.
        csv_path (str): Path to CSV file.
        models (List[str]): List of model names to compare (e.g., ['knn', 'svm', 'rf', 'gbc']).
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of metrics for each model.
    """
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

    results = {}
    if 'knn' in models:
        clf_knn = neighbors.KNeighborsClassifier(15, weights='distance')
        clf_knn.fit(X_train, y_train)
        results['KNN'] = {
            'Train Acc': accuracy_score(y_train, clf_knn.predict(X_train)),
            'Test Acc': accuracy_score(y_test, clf_knn.predict(X_test)),
            'Kappa (Test)': cohen_kappa_score(y_test, clf_knn.predict(X_test))
        }
    if 'svm' in models:
        clf_svm = svm.SVC(kernel='rbf', probability=True, gamma=0.5, decision_function_shape='ovo')
        clf_svm.fit(X_train, y_train)
        results['SVM'] = {
            'Train Acc': accuracy_score(y_train, clf_svm.predict(X_train)),
            'Test Acc': accuracy_score(y_test, clf_svm.predict(X_test)),
            'Kappa (Test)': cohen_kappa_score(y_test, clf_svm.predict(X_test))
        }
    if 'rf' in models:
        clf_rf = RandomForestClassifier(45)
        clf_rf.fit(X_train, y_train)
        results['RF'] = {
            'Train Acc': accuracy_score(y_train, clf_rf.predict(X_train)),
            'Test Acc': accuracy_score(y_test, clf_rf.predict(X_test)),
            'Kappa (Test)': cohen_kappa_score(y_test, clf_rf.predict(X_test))
        }
    if 'gbc' in models:
        clf_gbc = GradientBoostingClassifier(n_estimators=440)
        clf_gbc.fit(X_train, y_train)
        results['GBC'] = {
            'Train Acc': accuracy_score(y_train, clf_gbc.predict(X_train)),
            'Test Acc': accuracy_score(y_test, clf_gbc.predict(X_test)),
            'Kappa (Test)': cohen_kappa_score(y_test, clf_gbc.predict(X_test))
        }
    return results

def print_comparison_table(results_by_delta: Dict[float, Dict[str, Dict[str, float]]], models: List[str]) -> None:
    """
    Print a markdown table comparing accuracy and kappa for each model and delta.
    """
    headers = ['Delta'] + [f'{model} Train Acc' for model in models] + [f'{model} Test Acc' for model in models] + [f'{model} Kappa (Test)' for model in models]
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '---|' * len(headers))
    for delta, res in results_by_delta.items():
        row = [str(delta)]
        for model in models:
            key = model.upper() if model != 'rf' else 'RF'
            row.append(f"{res.get(key, {}).get('Train Acc', ''):.3f}" if key in res else '')
        for model in models:
            key = model.upper() if model != 'rf' else 'RF'
            row.append(f"{res.get(key, {}).get('Test Acc', ''):.3f}" if key in res else '')
        for model in models:
            key = model.upper() if model != 'rf' else 'RF'
            row.append(f"{res.get(key, {}).get('Kappa (Test)', ''):.3f}" if key in res else '')
        print('| ' + ' | '.join(row) + ' |')

def main():
    csv_path = 'CentralWater_20211230_comb.csv'
    deltas = [1, 10, 100, 1000, 10000]
    models = ['knn', 'svm', 'rf', 'gbc']
    results_by_delta = {}
    for delta in deltas:
        results = acc_com_by_bh(delta, csv_path, models)
        results_by_delta[delta] = results
    print_comparison_table(results_by_delta, models)

if __name__ == "__main__":
    main()
