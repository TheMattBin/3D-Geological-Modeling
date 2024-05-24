import pandas as pd
import numpy as np
from collections import defaultdict


sptdata = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\sptgeo_comb.csv')
sptdata.drop(['BH_Type', 'Report No.', 'Hole ID'], axis=1, inplace=True)
sptdata.sort_values(by=['Easting_x','Northing_x'], inplace=True)

#cptdata = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\CPT.csv')
cptdata = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\CPT110.csv')
cptdata.sort_values(by=['Easting','Northing'], inplace=True)

cpt_elev = defaultdict(list)
l = len(cptdata['Easting'])
spt_elev = defaultdict(list)
qc_2_append = defaultdict(list)
spt_df = []

for i in range(l):
    cpt_elev[(cptdata['Easting'][i], cptdata['Northing'][i])].append(cptdata['ElevCPT'][i])

for key in cpt_elev.keys():
    tmpdf = sptdata[((sptdata['Easting_x']-key[0])**2 + (sptdata['Northing_x']-key[1])**2) <= 12100]
    spt_df.append(tmpdf)
    for i, val in tmpdf.iterrows():
        spt_elev[(val['Easting_x'], val['Northing_x'])].append([val['TopElev_y'], val['BotElev_y']])


for k, value in spt_elev.items():
    tmpdf = cptdata[((cptdata['Easting'] - k[0]) ** 2 + (cptdata['Northing'] - k[1]) ** 2) <= 12100]
    for v in value:
        tmp_av = tmpdf[(v[1] <= tmpdf['ElevCPT']) & (tmpdf['ElevCPT'] <= v[0])]
        print(tmp_av)
        ave_qc = tmp_av['Qc'].mean(axis=0)
        qc_2_append[k].append(ave_qc)


east = [[k[0]] * len(v) for k, v in qc_2_append.items()]
north = [[k[1]] * len(v) for k, v in qc_2_append.items()]

dfqc = pd.DataFrame(
    {'Easting': [ee for e in east for ee in e], 'Northing': [nn for n in north for nn in n],
     'qc': [vv for k, v in qc_2_append.items() for vv in v]})
#dfqc.sort_values(by=['Easting','Northing'], inplace=True)
print(dfqc)

spt_df = pd.concat(spt_df, sort=False)
#spt_df.sort_values(by=['Easting_x','Northing_x'], inplace=True)

spt_df['East'] = [ee for e in east for ee in e]
spt_df['North'] = [nn for n in north for nn in n]
spt_df['qc'] = [vv for k, v in qc_2_append.items() for vv in v]
print(spt_df)

#CPT_SPT_joined = sptdata.merge(dfqc, left_on='Easting_x', right_on='Easting')
#spt_df.to_csv('trial_cpt_spt_110.csv')

cptdata25 = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\trial_cpt_spt_25.csv')
cptdata50 = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\trial_cpt_spt_50.csv')
cptdata80 = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\trial_cpt_spt_80.csv')
cptdata90 = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\trial_cpt_spt_90.csv')
cptdata95 = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\trial_cpt_spt_95.csv')
cptdata100 = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\trial_cpt_spt_100.csv')
cptdata110 = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\trial_cpt_spt_110.csv')

combdf = pd.concat([cptdata25, cptdata50, cptdata80, cptdata90, cptdata95, cptdata100, cptdata110], sort=False)
combdf.to_csv('combine_cpt_spt.csv')