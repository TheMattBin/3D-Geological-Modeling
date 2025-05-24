import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors
import netCDF4 as nc

def compare_borehole_logs(nc_path: str, csv_path: str) -> None:
    var = nc.Dataset(nc_path)
    df = pd.read_csv(csv_path)
    code = defaultdict(list)
    for i in range(len(df['Easting'])):
        code[(df['Easting'][i], df['Northing'][i])].append(df['Legend Code'][i])
    litoMatrix_knn = var['Lithology'][:]
    EntropyMatrix_knn = var['Information Entropy'][:]
    Xknn = var['x'][:]
    Yknn = var['y'][:]
    Zknn = var['z'][:]
    predict_bh_list = []
    ori_bh_list = []
    for i, val in code.items():
        idx_east_knn = (np.abs(Xknn - i[0])).argmin()
        idx_north_knn = (np.abs(Yknn - i[1])).argmin()
        predict_bh = litoMatrix_knn[:, idx_north_knn, idx_east_knn][9:-14]
        predict_bh = np.reshape(predict_bh, (-1, 1))
        predict_bh_list.append(predict_bh)
        ori_bh_list.append(np.reshape(np.array(val), (-1, 1)))
    fig, ax = plt.subplots(nrows=1, ncols=16, sharex=True, sharey=True)
    colors = ['#ff8000', '#ffa54d', '#ffff00']
    cmap = matplotlib.colors.ListedColormap(colors)
    for i in range(len(predict_bh_list)):
        ax[(i % 8 + i)].imshow(predict_bh_list[i], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
        ax[(i % 8 + i + 1)].imshow(ori_bh_list[i], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
        ax[(i % 8 + i)].get_yaxis().set_visible(False)
        ax[(i % 8 + i + 1)].get_yaxis().set_visible(False)
        ax[(i % 8 + i)].get_xaxis().set_visible(False)
        ax[(i % 8 + i + 1)].get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

def main():
    nc_path = 'kNN_airport_250x250x2_GeoTech100 distance.nc'
    csv_path = 'Model Comparison/comp2.csv'
    compare_borehole_logs(nc_path, csv_path)

if __name__ == "__main__":
    main()
