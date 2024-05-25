import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from collections import defaultdict
import os
import matplotlib.colors
import collections
from matplotlib.patches import Ellipse, Polygon

var = nc.Dataset(r'D:\pythonProject\3D Modelling\CPT Model\kNN_airport_250x250x2_GeoTech100 distance.nc')
df = pd.read_csv(r'D:\pythonProject\3D Modelling\Model Comparison\comp2.csv')

code = defaultdict(list)
l = len(df['Easting'])
for i in range(l):
    code[(df['Easting'][i], df['Northing'][i])].append(df['Legend Code'][i])


# getting dimensions and soil layers

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
    predict_bh = np.reshape(predict_bh, (-1,1))
    predict_bh_list.append(predict_bh)
    ori_bh_list.append(np.reshape(np.array(val),(-1,1)))

'''
for i in range(0, len(predict_bh_list), 4):
    fig, ax = plt.subplots(nrows=1, ncols=8, sharex=True, sharey=True)
    colors = ['#FFFF00', '#FFC000', '#FF0000']
    for j in range(i, i+4):
        cmap = matplotlib.colors.ListedColormap(colors)
        ax[(j%4+j)%8].imshow(predict_bh_list[j], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
        ax[(j%4+j+1)%8].imshow(ori_bh_list[j], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
        ax[(j%4+j)%8].get_yaxis().set_visible(False)
    plt.tight_layout()
    #plt.show()
'''
fig, ax = plt.subplots(nrows=1, ncols=16, sharex=True, sharey=True)
colors = ['#ff8000', '#ffa54d', '#ffff00']
cmap = matplotlib.colors.ListedColormap(colors)
for i in range(len(predict_bh_list)):
    ax[(i%8+i)].imshow(predict_bh_list[i], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
    ax[(i%8+i+1)].imshow(ori_bh_list[i], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
    ax[(i%8+i)].get_yaxis().set_visible(False)
    ax[(i % 8 + i + 1)].get_yaxis().set_visible(False)
    ax[(i % 8 + i)].get_xaxis().set_visible(False)
    ax[(i % 8 + i+1)].get_xaxis().set_visible(False)
plt.tight_layout()
#plt.show()


fig, ax = plt.subplots(nrows=1, ncols=16, sharex=True, sharey=True)
colors = ['#ff8000', '#ffa54d', '#ffff00']
cmap = matplotlib.colors.ListedColormap(colors)
ax[0].imshow(predict_bh_list[4], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
ax[0].get_xaxis().set_visible(False)
#ax[0].add_patch(Polygon(0,1,hatch='\\/...', facecolor='g'))
ax[0].set_ylabel('Depth below seabed (m)')

ax[1].imshow(ori_bh_list[4], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)

ax[2].imshow(predict_bh_list[5], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)

ax[3].imshow(ori_bh_list[5], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
ax[3].get_xaxis().set_visible(False)
ax[3].get_yaxis().set_visible(False)

ax[4].imshow(predict_bh_list[6], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
ax[4].get_xaxis().set_visible(False)
ax[4].get_yaxis().set_visible(False)

ax[5].imshow(ori_bh_list[6], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
ax[5].get_xaxis().set_visible(False)
ax[5].get_yaxis().set_visible(False)

ax[6].imshow(predict_bh_list[7], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
ax[6].get_xaxis().set_visible(False)
ax[6].get_yaxis().set_visible(False)

ax[7].imshow(ori_bh_list[7], aspect='auto', cmap=cmap, extent=[0, 1, 40, 0])
ax[7].get_xaxis().set_visible(False)
ax[7].get_yaxis().set_visible(False)



plt.tight_layout()
plt.show()



#df = pd.read_csv(r'D:\pythonProject\3D Modelling\CPT Model\airport_comp.csv')
'''
code = defaultdict(list)
coor_east = defaultdict(list)
coor_north = defaultdict(list)
topE = defaultdict(list)
botE = defaultdict(list)

l = len(df['Location ID'])
for i in range(l):
    code[df['Location ID'][i]].append([df['Code'][i]] * df['Count'][i])
    coor_east[df['Location ID'][i]].append([df['Easting'][i]] * df['Count'][i])
    coor_north[df['Location ID'][i]].append([df['Northing'][i]] * df['Count'][i])
    topE[df['Location ID'][i]].append([df['TopElev'][i]] * df['Count'][i])
    botE[df['Location ID'][i]].append([df['BotElev'][i]] * df['Count'][i])
print(code)
print(coor_north)
df2 = pd.DataFrame({'Easting': [k for i in coor_east.values() for j in i for k in j], 'Northing': [k for i in coor_north.values() for j in i for k in j],
                    'TopElev': [k for i in topE.values() for j in i for k in j], 'BotElev': [k for i in botE.values() for j in i for k in j],
                    'Legend Code': [k for i in code.values() for j in i for k in j]})
df2.to_csv('comp.csv')
'''
