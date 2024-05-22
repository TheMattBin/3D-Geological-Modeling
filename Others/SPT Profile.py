import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches

def SPTGraph(spt, top, key):
    plt.plot(spt, top, 'k-')
    plt.ylabel('Depth (m)')
    plt.xlabel('SPT-N (blows/m)')
    plt.title('Depth against SPT-N Graph of ' + key)
    plt.axis([0, max(spt), max(top), 0])
    #plt.gca().invert_yaxis()
    plt.savefig(r'D:\HK Model Arcgis\HK Model ArcPro\Scripts\SPT Img\\' + key + ".png")
    #plt.show()
    plt.clf()

df = pd.read_csv(r'D:\HK Model Arcgis\HK Model ArcPro\trail2\SPT 450.csv')
print(df.head())
depth = defaultdict(list)
SPTN = defaultdict(list)

l = len(df['Location ID'])
for i in range(l):
    depth[df['Location ID'][i]].append(df['Depth'][i])
for i in range(l):
    SPTN[df['Location ID'][i]].append(df['N2'][i])


for key in depth.keys():
    spt_tmp = list(SPTN[key])
    dep_tmp = list(depth[key])
    if '/' in key:
        pass
    else:
        if len(spt_tmp) >=2:
            SPTGraph(spt_tmp, dep_tmp, key)