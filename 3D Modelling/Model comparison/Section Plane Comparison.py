import rasterio
from sklearn import neighbors
import netCDF4
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
import shapefile
from shapely.geometry import LineString, MultiPoint
from shapely.ops import split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from collections import defaultdict
import os
import matplotlib.colors
import collections


'''
shape = shapefile.Reader(r"D:\HK Model Arcgis\HK Model ArcPro\Demo_21Sept\HK Model 21 Sept\Section Peng Chau.shp")
feature = shape.shapeRecords()[0]
first = feature.shape.__geo_interface__
line = first['coordinates']
print(line)
for i in range(len(line)-1):
    dis = np.sqrt((line[i][0] - line[i+1][0]) ** 2 + (line[i][1]-line[i+1][1])**2)
    segment = int(dis//50)
    a = line[i]
    b = line[i+1]
    print(a,b)
    line2 = LineString([a,b])
    print(line2)
    splitter = MultiPoint([line2.interpolate((j / segment)) for j in range(segment)])
'''

ds_knn100_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNN_pc_nohide_10x10x02_100 distance.nc')
ds_knn1_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNN_pc_nohide_10x10x02_1 distance.nc')
ds_knn10000_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model kNN\kNN_pc_nohide_10x10x02_10000 distance.nc')

ds_gbc1_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model Gradient Boosting\myOutputGBC-pengchau1.nc')
ds_gbc100_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model Gradient Boosting\myOutputGBC-pengchau100.nc')
ds_gbc10000_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model Gradient Boosting\myOutputGBC-pengchau10000.nc')

ds_rf1_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model RF\myOutputRF-PengChau-1.nc')
ds_rf100_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model RF\myOutputRF-PengChau-100.nc')
ds_rf10000_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model RF\myOutputRF-PengChau-10000.nc')

ds_svm1_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model SVM\myOutputSVM-PC-1.nc')
ds_svm100_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model SVM\myOutputSVM-PC-100.nc')
ds_svm1000_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model SVM\myOutputSVM-PC-1000.nc')
ds_svm10000_pc = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model SVM\myOutputSVM-PC-10000.nc')


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
    key_list = []
    predict_list_knn_entropy = []

    df = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\pt_along_pc.csv')
    east = list(df['Easting'].values)
    print(east)
    north = list(df['Northing'].values)


    def BHvsPre(var, east, north):
        # getting dimensions and soil layers
        litoMatrix_knn = var['Lithology'][:]
        EntropyMatrix_knn = var['Information Entropy'][:]
        Xknn = var['x'][:]
        Yknn = var['y'][:]
        Zknn = var['z'][:]
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


    for e,n in zip(east, north):
        BHvsPre(ds_svm, e, n)


    topelev = []
    botelev = []

    for i in depth_knn:
        topelev.append(list(i[:]+0.5))
        tmp = list(i[1:] + 0.5)
        bot = i[-1] -0.5
        botelev.append(tmp + [bot])

    df2 = pd.DataFrame({'Easting': [j for i in coor_df_east for j in i], 'Northing': [j for i in coor_df_north for j in i],
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

#for i in [ds_svm1_pc, ds_svm100_pc, ds_svm10000_pc]:
#    predictedBH(i)

knn_df100_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\kNN_pc_nohide_10x10x02_1 distance_test.csv')
knn_df1_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\kNN_pc_nohide_10x10x02_100 distance_test.csv')
knn_df10000_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\kNN_pc_nohide_10x10x02_10000 distance_test.csv')

gbc_df1_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputGBC-pengchau1_test.csv')
gbc_df100_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputGBC-pengchau100_test.csv')
gbc_df10000_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputGBC-pengchau10000_test.csv')

rf_df1_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputRF-PengChau-1_test.csv')
rf_df100_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputRF-PengChau-100_test.csv')
rf_df10000_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputRF-PengChau-10000_test.csv')

svm_df1_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputSVM-PC-1_test.csv')
svm_df100_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputSVM-PC-100_test.csv')
svm_df1000_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputSVM-PC-1000_test.csv')
svm_df10000_pc_nh = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\Section West Peng Chau Compare\myOutputSVM-PC-10000_test.csv')

section_tol = []
entropy_tol = []

def XSectionCom(df):
    code = defaultdict(list)
    entropy = defaultdict(list)
    east_x = defaultdict(list)
    north_y = defaultdict(list)
    l = len(df['Easting'])

    for i in range(l):
        code[(df['Easting'][i], df['Northing'][i])].append(df['Legend Code'][i])
        entropy[(df['Easting'][i], df['Northing'][i])].append(float(df['Information Entropy'][i]))
    print(code)


    section_plane = []
    for key, value in code.items():
        section_plane.append(code[key][100:])
    #print(section_plane)

    entropy_plane = []
    for key, value in entropy.items():
        entropy_plane.append(entropy[key][100:])
    print(entropy_plane)

    section_plane_trans = np.transpose(np.array(section_plane))
    section_tol.append(section_plane_trans)
    entropy_plane_trans = np.transpose(np.array(entropy_plane))
    entropy_tol.append(entropy_plane_trans)
    #en_color = plt.imshow(entropy_plane_trans, aspect='auto', cmap=plt.cm.get_cmap('jet'))
    #plt.colorbar(orientation='horizontal')
    #cbar = plt.colorbar(en_color)
    #plt.show()

for i in [svm_df100_pc_nh, svm_df1000_pc_nh, svm_df10000_pc_nh]:
    XSectionCom(i)

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
colors = ['#002060', '#0070C0', '#66FFFF', '#FFFF00', '#FFC000', '#FF0000']
cmap = matplotlib.colors.ListedColormap(colors)
ax1 = ax[0].imshow(section_tol[0], aspect='auto', cmap=cmap, extent=[0, 1750, -42, -4])
ax100 = ax[1].imshow(section_tol[1], aspect='auto', cmap=cmap, extent=[0, 1750, -42, -4])
ax10000 = ax[2].imshow(section_tol[2], aspect='auto', cmap=cmap, extent=[0, 1750, -42, -4])
ax100_entro = ax[3].imshow(entropy_tol[1], aspect='auto', cmap=cmap, extent=[0, 1750, -42, -4])
plt.xlabel('Distance (m)')
plt.ylabel('Elevation (m)')
plt.tight_layout()
plt.show()

