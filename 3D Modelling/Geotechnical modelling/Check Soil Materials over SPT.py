import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from collections import defaultdict
import os
import matplotlib.colors
import collections

ds_knn_airport = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNN_airport_250x250x2_100 distance.nc')

def predicted_soil_along_spt(var):
    ds_svm = var
    p = var.filepath()
    name = os.path.basename(p)
    name = os.path.splitext(name)
    elev = defaultdict(list)
    sptn = defaultdict(list)
    coor = defaultdict(list)
    coor_df_east = []
    coor_df_north = []
    predict_list = []
    spt_list = []
    key_list = []
    midelev = []

    df = pd.read_csv(r'D:\HK Model Arcgis\HK Model ArcPro\Boreline Trial\airport.csv')
    l = len(df['Rep_BH_ID'])

    for i in range(l):
        elev[df['Rep_BH_ID'][i]].append([df['TopElev'][i], df['BotElev'][i], df['MidElev'][i]])
    for i in range(l):
        sptn[df['Rep_BH_ID'][i]].append(df['SPTN'][i])
    for i in range(l):
        coor[df['Rep_BH_ID'][i]] = ([df['Easting'][i], df['Northing'][i]])


    def BHvsPre(var, east, north, depth, sptn):
        # getting dimensions and soil layers
        litoMatrix = var['Lithology'][:]
        EntropyMatrix = var['Information Entropy'][:]
        X = var['x'][:]
        Y = var['y'][:]
        Z = var['z'][:]
        idx_east = (np.abs(X - east)).argmin()
        idx_north = (np.abs(Y - north)).argmin()

        predict_bh = litoMatrix[:, idx_north, idx_east]
        predict_entropy = EntropyMatrix[:, idx_north, idx_east]

        if predict_bh.size != 0:
            depth.append(var['z'][:])
            predict_list.append(predict_bh)
            coor_df_east.append([east] * (len(predict_bh)))
            coor_df_north.append([north] * (len(predict_bh)))
            key_list.append([key] * (len(predict_bh)))
        spt_list.append(sptn)


    for key in elev.keys():
        BHvsPre(ds_svm, coor[key][0], coor[key][1], midelev, sptn[key])

    df2 = pd.DataFrame(
        {'Location ID': [j for i in key_list for j in i], 'Easting': [j for i in coor_df_east for j in i],
         'Northing': [j for i in coor_df_north for j in i], 'MidElev': [j for i in midelev for j in i ], 'Legend Code': [j for i in predict_list for j in i ]})

    print(df2)
    df2.to_csv('{}_test.csv'.format(name[0]))

predicted_soil_along_spt(ds_knn_airport)