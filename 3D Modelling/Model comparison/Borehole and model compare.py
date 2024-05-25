import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from collections import defaultdict
import os
import matplotlib.colors
import collections

# XSection_from_3DModel(ds, 821926.48, 815848.66, -6.1, -6.1, -74.43, [3,3,3,5,5,5,5,5,5,5,5,9,9,9,9,9,10])

# kNN
# -20.12, -20.12, -148.63, [3,3,5,5,5,5,9,9,10,9,10,9,10,10,10,10,10,10,10,10,10] R_18325 H_II/MD-LQ/T3
# -5.3, -5.3, -85.97, [3,3,3,5,5,3,5,5,5,5,10,10] R_54725-54726 H_BH 7
# -5.15, -5.15, -85.54, [3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5,5,10,10,10]
# R_54725-54726 H_BH 3 -4.9, -4.9, -85.13, [3,3,3,5,5,9,9,10,10,9,10,10,10,9,10,10]
# R_39294 H_MBH11 -7.25, -7.25, -84.4, [3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5,9,9,9,9,9,9,9,10]
# R_11729 H_GIS3 -6.1, -6.1, -74.43, [3,3,3,5,5,5,5,5,5,5,5,9,9,9,9,9,10]

ds_knn1 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNNOutput500x500x1_1 distance.nc')
ds_knn10 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNNOutput500x500x1_10 distance.nc')
ds_knn100 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNNOutput500x500x1_100 distance.nc')
ds_knn1000 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNNOutput500x500x1_1000 distance.nc')
ds_knn10000 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNNOutput500x500x1_10000 distance.nc')

ds_svm1 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model SVM\myOutputSVM1000-08-03-2022-20-44-51.nc')
ds_svm10 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model SVM\myOutputSVM1000-08-03-2022-21-04-13.nc')
ds_svm100 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model SVM\myOutputSVM1000-08-03-2022-21-19-20.nc')
ds_svm1000 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model SVM\myOutputSVM1000-08-03-2022-21-26-37.nc')
ds_svm10000 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model SVM\myOutputSVM1000-08-03-2022-21-33-58.nc')

ds_knn1_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNN_pc_50x50x1_1 distance.nc')
ds_knn100_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNN_pc_50x50x1_100 distance.nc')
ds_knn10000_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNN_pc_50x50x1_10000 distance.nc')

ds_knn100_pc_nh = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNN_pc_nohide_50x50x1_100 distance.nc')

def predictedBH(var):
    ds_svm = var
    p = var.filepath()
    name = os.path.basename(p)
    name = os.path.splitext(name)
    depth = defaultdict(list)
    depth_knn = []
    code = defaultdict(list)
    coor = defaultdict(list)
    coor_df_east = []
    coor_df_north = []
    predict_list_knn = []
    soil_list = []
    key_list = []
    predict_list_knn_entropy = []

    df = pd.read_csv(r'D:\pythonProject\BH Data Processing\CentralWater_20211230_comb.csv')
    l = len(df['Location ID'])

    for i in range(l):
        depth[df['Location ID'][i]].append(df['Ground Level'][i] - df['Depth Top'][i])
    for i in range(l):
        depth[df['Location ID'][i]].append(df['Ground Level'][i] - df['Depth Base'][i])
    for i in range(l):
        code[df['Location ID'][i]].append(df['Legend Code'][i])
    for i in range(l):
        coor[df['Location ID'][i]] = ([df['Easting'][i], df['Northing'][i]])


    def BHvsPre(var, east, north, depth, depth_t, depth_b, soil, key):
        # getting dimensions and soil layers
        litoMatrix_knn = var['Lithology'][:]
        EntropyMatrix_knn = var['Information Entropy'][:]
        Xknn = var['x'][:]
        Yknn = var['y'][:]
        Zknn = var['z'][:]
        idx_depth_knn = (np.abs(Zknn - depth)).argmin()
        idx_depth_top_knn = (np.abs(Zknn - depth_t)).argmin()
        idx_depth_bot_knn = (np.abs(Zknn - depth_b)).argmin()
        idx_east_knn = (np.abs(Xknn - east)).argmin()
        idx_north_knn = (np.abs(Yknn - north)).argmin()

        predict_bh_knn = litoMatrix_knn[:, idx_north_knn, idx_east_knn]
        predict_entropy_knn = EntropyMatrix_knn[:, idx_north_knn, idx_east_knn]
        if predict_bh_knn.size != 0:
            depth_knn.append(var['z'][:])
            predict_list_knn.append(predict_bh_knn)
            predict_list_knn_entropy.append(predict_entropy_knn)
            coor_df_east.append([east] * (len(predict_bh_knn)))
            coor_df_north.append([north] * (len(predict_bh_knn)))
            key_list.append([key]*(len(predict_bh_knn)))
        soil_list.append(soil)


    for key in depth.keys():
        mxi = max(depth[key])
        mini = min(depth[key])
        BHvsPre(ds_svm, coor[key][0], coor[key][1], mxi, mxi, mini, code[key], key)


    topelev = []
    botelev = []

    for i in depth_knn:
        topelev.append(list(i[:]+0.5))
        tmp = list(i[1:] + 0.5)
        bot = i[-1] -0.5
        botelev.append(tmp + [bot])

    df2 = pd.DataFrame({'Location ID': [j for i in key_list for j in i], 'Easting': [j for i in coor_df_east for j in i], 'Northing': [j for i in coor_df_north for j in i],
                        'TopElev': [j for i in topelev for j in i ], 'BotElev': [j for i in botelev for j in i ], 'Legend Code': [j for i in predict_list_knn for j in i ],
                        'Information Entropy': [j for i in predict_list_knn_entropy for j in i ]})
    df2['Lithology'] = ''
    df2['Ground Level'] = ''
    df2['Final Depth'] = ''

    for i, v in df2.iterrows():
        if v['Legend Code'] == 0:
            df2.loc[i, 'Lithology'] = 'Water'
        elif v['Legend Code'] == 1:
            df2.loc[i, 'Lithology'] = 'Fill'
        elif v['Legend Code'] == 2:
            df2.loc[i, 'Lithology'] = 'Marine deposit'
        elif v['Legend Code'] == 3:
            df2.loc[i, 'Lithology'] = 'Alluvium'
        elif v['Legend Code'] == 4:
            df2.loc[i, 'Lithology'] = 'Grade V-IV rocks'
        else:
            df2.loc[i, 'Lithology'] = 'Grade III-I rocks'

    print(df2)
    df2.to_csv('{}_test.csv'.format(name[0]))

#for i in [ds_knn100_pc_nh]:
#    predictedBH(i)

df0 = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section_Compare.csv')
section_plane_ori = set(df0['Location ID'])
df1 = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\kNNOutput500x500x1_1 distance_test.csv')
df10 = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\kNNOutput500x500x1_10 distance_test.csv')
df100 = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\kNNOutput500x500x1_100 distance_test.csv')
df10000 = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\kNNOutput500x500x1_10000 distance_test.csv')

df1_pc = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\kNN_pc_50x50x1_1 distance_test.csv')
df100_pc = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\kNN_pc_50x50x1_100 distance_test.csv')
df10000_pc = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\kNN_pc_50x50x1_10000 distance_test.csv')

df100_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\kNN_pc_nohide_50x50x1_100 distance_test.csv')

def XSectionCom(df):
    reflist2 = ['R_3572 H_MC36', 'R_38469 H_MBH2', 'R_2412 H_PC301', 'R_37353 H_DH1', 'R_25765 H_TL 1',
               'R_37353 H_DH2', 'R_25765 H_TL 3', 'R_25765 H_TL 2', 'R_37353 H_DH3', '	R_37353 H_DH6',
               'R_2412 H_PC303', 'R_37353 H_DH5', 'R_2412 H_PC304', 'R_2412 H_PC305', 'R_2412 H_PC307', 'R_2412 H_PC309',
               'R_2412 H_PC310', 'R_2412 H_PC314', 'R_2412 H_PC313', 'R_2412 H_PC315', 'R_2412 H_PC316', 'R_49001 H_BH3', 'R_49001 H_BH4',
               'R_2412 H_PC317', 'R_49001 H_BH2', 'R_49001 H_BH1', 'R_2412 H_PC318', 'R_25940 H_MS14', 'R_2412 H_PC319',
               'R_25940 H_MS13', 'R_25940 H_BH16', 'R_25940 H_MS12', 'R_2412 H_PC320', 'R_25940 H_MS11',
               'R_25940 H_BH15', 'R_25940 H_MS10', 'R_25940 H_MS9', 'R_2412 H_PC322', 'R_25940 H_BH13', 'R_25940 H_BH12',
               'R_2412 H_PC323', 'R_2412 H_PC324', 'R_32389 H_PS 2', 'R_2412 H_PC325', 'R_32389 H_PS 1', 'R_2412 H_PC326',
               'R_2412 H_PC327', 'R_2412 H_PC328', 'R_2412 H_PC329', 'R_2412 H_PC330', 'R_2412 H_PC331', 'R_2412 H_PC332',
               'R_2412 H_PC333', 'R_2412 H_PC334', 'R_2412 H_PC335', 'R_2412 H_PC336', 'R_2412 H_PC337', 'R_2412 H_PC338',
               'R_2412 H_PC339', 'R_32005 H_B4', 'R_32005 H_B6', 'R_32654 H_B 2', 'R_32654 H_B 1', 'R_29358 H_DH 4', 'R_32654 H_B 5',
               'R_32654 H_B 3', 'R_32654 H_B 7', 'R_42887 H_BH-10', 'R_42887 H_BH-11', 'R_42887 H_BH-09',
               'R_20517 H_BH-08', 'R_42887 H_BH-01', 'R_42887 H_BH-02', 'R_42887 H_BH-03', 'R_42887 H_BH-04',
               'R_20501 H_BH6', 'R_20501 H_BH7']

    reflist = ['R_3572 H_MC36', 'R_38469 H_MBH2', 'R_37353 H_DH5', 'R_2412 H_PC309', 'R_2412 H_PC310',
               'R_2412 H_PC313', 'R_2412 H_PC315', 'R_25940 H_MS14', 'R_25940 H_MS9', 'R_2412 H_PC326',
               'R_2412 H_PC330', 'R_2412 H_PC338', 'R_19722 H_MS-01', 'R_42887 H_BH-09', 'R_20517 H_BH-08', 'R_42887 H_BH-01',
               'R_42887 H_BH-02', 'R_42887 H_BH-03', 'R_42887 H_BH-04', 'R_20501 H_BH6', 'R_20501 H_BH7']
    code = defaultdict(list)
    entropy = defaultdict(list)
    east_x = defaultdict(list)
    north_y = defaultdict(list)
    l = len(df['Location ID'])

    for i in range(l):
        if df['Location ID'][i] in reflist2:
            code[df['Location ID'][i]].append(df['Legend Code'][i])
            entropy[df['Location ID'][i]].append(float(df['Information Entropy'][i]))
            east_x[df['Location ID'][i]].append(df['Easting'][i])
            north_y[df['Location ID'][i]].append(df['Northing'][i])


    df_pt = pd.DataFrame({'Easting': [j for i in east_x.values() for j in i], 'Northing': [j for i in north_y.values() for j in i]})
    df_pt.to_csv('pt_check.csv')

    #print(code)
    code2 = collections.OrderedDict(sorted(code.items(), key=lambda pair: reflist2.index(pair[0])))
    entropy2 = collections.OrderedDict(sorted(entropy.items(), key=lambda pair: reflist2.index(pair[0])))

    section_plane = []
    for key, value in code2.items():
        if key in section_plane_ori:
            section_plane.append(code2[key][20:])
    #print(section_plane)

    entropy_plane = []
    for key, value in entropy2.items():
        if key in section_plane_ori:
            entropy_plane.append(entropy2[key][20:])
    print(entropy_plane)

    section_plane_trans = np.transpose(np.array(section_plane))
    entropy_plane_trans = np.transpose(np.array(entropy_plane))
    colors = ['#002060', '#0070C0', '#66FFFF', '#FFFF00', '#FFC000', '#FF0000']
    cmap = matplotlib.colors.ListedColormap(colors)
    #plt.imshow(section_plane_trans, aspect='auto', cmap=cmap)
    en_color = plt.imshow(entropy_plane_trans, aspect='auto', cmap=plt.cm.get_cmap('jet'))
    plt.colorbar(orientation='horizontal')
    cbar = plt.colorbar(en_color)
    plt.show()
XSectionCom(df100_pc_nh)

# constrained_layout=True

def XSection_from_3DModel(soil, predict_bh_knn, predict_bh_svm):
    l = len(predict_bh_svm)
    print(l)
    for i in range(0, l, 10):
        fig, ax = plt.subplots(nrows=3, ncols=10)
        cluster0 = np.repeat(np.expand_dims(soil[i], 1), 1, 1)
        cluster_t0 = np.repeat(np.expand_dims(predict_bh_knn[i], 1), 1, 1)
        cluster_t_svm0 = np.repeat(np.expand_dims(predict_bh_svm[i], 1), 1, 1)
        cluster1 = np.repeat(np.expand_dims(soil[i + 1], 1), 1, 1)
        cluster_t1 = np.repeat(np.expand_dims(predict_bh_knn[i + 1], 1), 1, 1)
        cluster_t_svm1 = np.repeat(np.expand_dims(predict_bh_svm[i + 1], 1), 1, 1)
        cluster2 = np.repeat(np.expand_dims(soil[i + 2], 1), 1, 1)
        cluster_t2 = np.repeat(np.expand_dims(predict_bh_knn[i + 2], 1), 1, 1)
        cluster_t_svm2 = np.repeat(np.expand_dims(predict_bh_svm[i + 2], 1), 1, 1)
        cluster3 = np.repeat(np.expand_dims(soil[i + 3], 1), 1, 1)
        cluster_t3 = np.repeat(np.expand_dims(predict_bh_knn[i + 3], 1), 1, 1)
        cluster_t_svm3 = np.repeat(np.expand_dims(predict_bh_svm[i + 3], 1), 1, 1)
        cluster4 = np.repeat(np.expand_dims(soil[i + 4], 1), 1, 1)
        cluster_t4 = np.repeat(np.expand_dims(predict_bh_knn[i + 4], 1), 1, 1)
        cluster_t_svm4 = np.repeat(np.expand_dims(predict_bh_svm[i + 4], 1), 1, 1)
        cluster5 = np.repeat(np.expand_dims(soil[i + 5], 1), 1, 1)
        cluster_t5 = np.repeat(np.expand_dims(predict_bh_knn[i + 5], 1), 1, 1)
        cluster_t_svm5 = np.repeat(np.expand_dims(predict_bh_svm[i + 5], 1), 1, 1)
        cluster6 = np.repeat(np.expand_dims(soil[i + 6], 1), 1, 1)
        cluster_t6 = np.repeat(np.expand_dims(predict_bh_knn[i + 6], 1), 1, 1)
        cluster_t_svm6 = np.repeat(np.expand_dims(predict_bh_svm[i + 6], 1), 1, 1)
        cluster7 = np.repeat(np.expand_dims(soil[i + 7], 1), 1, 1)
        cluster_t7 = np.repeat(np.expand_dims(predict_bh_knn[i + 7], 1), 1, 1)
        cluster_t_svm7 = np.repeat(np.expand_dims(predict_bh_svm[i + 7], 1), 1, 1)
        cluster8 = np.repeat(np.expand_dims(soil[i + 8], 1), 1, 1)
        cluster_t8 = np.repeat(np.expand_dims(predict_bh_knn[i + 8], 1), 1, 1)
        cluster_t_svm8 = np.repeat(np.expand_dims(predict_bh_svm[i + 8], 1), 1, 1)
        cluster9 = np.repeat(np.expand_dims(soil[i + 9], 1), 1, 1)
        cluster_t9 = np.repeat(np.expand_dims(predict_bh_knn[i + 9], 1), 1, 1)
        cluster_t_svm9 = np.repeat(np.expand_dims(predict_bh_svm[i + 9], 1), 1, 1)
        # cluster10 = np.repeat(np.expand_dims(soil[0][i+10], 1), 1, 1)
        # cluster_t10 = np.repeat(np.expand_dims(predict_bh_knn[1][i+10], 1), 1, 1)
        # cluster_t_svm10 = np.repeat(np.expand_dims(predict_bh_svm[2][i+10], 1), 1, 1)

        ax00 = ax[0][(i % 10)].imshow(cluster0, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax01 = ax[0][(i % 10) + 1].imshow(cluster1, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax02 = ax[0][(i % 10) + 2].imshow(cluster2, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax03 = ax[0][(i % 10) + 3].imshow(cluster3, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax04 = ax[0][(i % 10) + 4].imshow(cluster4, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax05 = ax[0][(i % 10) + 5].imshow(cluster5, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax06 = ax[0][(i % 10) + 6].imshow(cluster6, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax07 = ax[0][(i % 10) + 7].imshow(cluster7, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax08 = ax[0][(i % 10) + 8].imshow(cluster8, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax09 = ax[0][(i % 10) + 9].imshow(cluster9, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax10 = ax[1][(i % 10)].imshow(cluster_t0, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax11 = ax[1][(i % 10) + 1].imshow(cluster_t1, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax12 = ax[1][(i % 10) + 2].imshow(cluster_t2, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax13 = ax[1][(i % 10) + 3].imshow(cluster_t3, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax14 = ax[1][(i % 10) + 4].imshow(cluster_t4, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax15 = ax[1][(i % 10) + 5].imshow(cluster_t5, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax16 = ax[1][(i % 10) + 6].imshow(cluster_t6, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax17 = ax[1][(i % 10) + 7].imshow(cluster_t7, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax18 = ax[1][(i % 10) + 8].imshow(cluster_t8, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax19 = ax[1][(i % 10) + 9].imshow(cluster_t9, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax20 = ax[2][(i % 10)].imshow(cluster_t_svm0, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        ax21 = ax[2][(i % 10) + 1].imshow(cluster_t_svm1, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax22 = ax[2][(i % 10) + 2].imshow(cluster_t_svm2, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax23 = ax[2][(i % 10) + 3].imshow(cluster_t_svm3, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax24 = ax[2][(i % 10) + 4].imshow(cluster_t_svm4, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax25 = ax[2][(i % 10) + 5].imshow(cluster_t_svm5, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax26 = ax[2][(i % 10) + 6].imshow(cluster_t_svm6, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax27 = ax[2][(i % 10) + 7].imshow(cluster_t_svm7, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax28 = ax[2][(i % 10) + 8].imshow(cluster_t_svm8, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        ax29 = ax[2][(i % 10) + 9].imshow(cluster_t_svm9, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0,
                                              vmax=11)
        # ax2 = ax[(i%4) + 1].imshow(cluster_t, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)

        # ax1 = ax[0].imshow(litoMatrix[idx_depth], extent=extentXY, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        # ax2 = ax[1].imshow(litoMatrix[:, idx_north, idx_east:idx_east+1], aspect=0.05, cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)
        # ax3 = ax[(i%4) *2 +1].imshow(cluster_t, aspect='auto', cmap=plt.cm.get_cmap('jet', 12), vmin=0, vmax=11)

        # ax[0].set_title('{}'.format(bh))
        # ax[0].set_xlabel('Easting')
        # ax[0].set_ylabel('Elevation')

        # ax[1].set_title('Prediction of {}'.format(bh))
        # ax[1].set_xlabel('Easting')
        # ax[1].set_ylabel('Elevation')

        # ax[2].set_title('Along Easting of {0:.0f} m'.format(north))
        # ax[2].set_xlabel('Northing')
        # ax[2].set_ylabel('Depth')

        # cbar = fig.colorbar(ax3, ax=ax[:], ticks=range(12))
        # tick_locs = (np.arange(12) + 0.5)*(12-1)/12
        # cbar.set_ticks(tick_locs)
        # cbar.set_ticklabels(['Water', 'Fill', 'Beach deposit', 'Marine deposit', 'Estuarine deposit', 'Alluvium',
        #    'Debris flow deposit', 'Colluvium', 'Residual soil', 'Grade V', 'Grade III', 'Others'])
        ax[0][(i % 10)].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[0][(i % 10) + 1].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[0][(i % 10) + 2].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[0][(i % 10) + 3].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[0][(i % 10) + 4].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[0][(i % 10) + 5].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[0][(i % 10) + 6].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[0][(i % 10) + 7].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[0][(i % 10) + 8].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[0][(i % 10) + 9].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10)].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10) + 1].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10) + 2].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10) + 3].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10) + 4].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10) + 5].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10) + 6].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10) + 7].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10) + 8].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[1][(i % 10) + 9].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10)].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10) + 1].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10) + 2].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10) + 3].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10) + 4].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10) + 5].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10) + 6].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10) + 7].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10) + 8].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax[2][(i % 10) + 9].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        plt.savefig(r'D:\pythonProject\3D Modelling\Model Comparison\bh compare500\\' + '{}'.format(key_list[i]) + '.jpg')
        # plt.show()
        fig.clear()

    plt.show()
    fig.clear()


#XSection_from_3DModel(soil_list, predict_list_knn, predict_list_svm)