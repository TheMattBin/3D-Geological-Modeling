import pandas as pd
import numpy as np

cptdf = pd.read_csv(r'D:\pythonProject\3D Modelling\CPT Model\CPTcomb.csv')
idx = []
for i in range(len(cptdf['Check'])):
    if cptdf['Check'][i] == 1:
        idx.append(i)
idx.append(len(cptdf['Check']))

df_sep = []
tmpset = []
for j in range(len(idx)-1):
    tmp =cptdf.iloc[idx[j]:idx[j+1]]
    tmp = tmp.dropna(subset=['STCN_FRES'])
    df_sep.append(tmp)

for k in range(0,len(df_sep),5):
    print(k)
    with pd.ExcelWriter('output_CPT_{}.xlsx'.format(k)) as writer:
        for w in range(k,k+5):
            df_sep[w].to_excel(writer, sheet_name='Sheet_name_{}'.format(w))