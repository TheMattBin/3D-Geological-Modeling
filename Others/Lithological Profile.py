import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches

def BH_plot(soil, depth, color_map, key):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,8))
    cluster = np.repeat(np.expand_dims(soil, 1), 1, 1)
    color_bar = ['white','darkgreen', 'skyblue', 'yellow', 'pink', 'red', 'blue', 'cyan', 'purple', 'orange', 'lightgreen', 'grey']
    cmap_facies = colors.ListedColormap(color_bar[0:len(color_bar)], 'indexed')
    ax.imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies,
              vmin=1, vmax=len(color_bar), extent=[0,1 ,np.max(depth),np.min(depth)])
    plt.tick_params(bottom=False, labelbottom=False)
    hands = []
    for k in color_map.keys():
        if color_map[k] in soil:
            col = color_bar[(color_map[k])-1]
            hands.append(mpatches.Patch(color=col, label=k))
    plt.legend(handles=hands, loc='best', fontsize=8)
    plt.savefig(r'D:\HK Model Arcgis\HK Model ArcPro\Scripts\Img\\' + key + ".png")
    #plt.show()

df = pd.read_csv(r'D:\HK Model Arcgis\Excel Python files\MHags_V6.csv')
depth = defaultdict(list)
code = defaultdict(list)
color_map = {}

l = len(df['Location ID'])

for i in range(l):
    depth[df['Location ID'][i]].append(df['Depth Top'][i])
for i in range(l):
    depth[df['Location ID'][i]].append(df['Depth Base'][i])
for i in range(l):
    code[df['Location ID'][i]].append(df['Legend Code'][i])
for i in range(l):
    color_map[df['Geology Code 2'][i]] = (df['Legend Code'][i])


for key in depth.keys():
    depth_tmp = list(set(depth[key]))
    if '/' in key:
        pass
    else:
        BH_plot(code[key], depth_tmp, color_map, key)
