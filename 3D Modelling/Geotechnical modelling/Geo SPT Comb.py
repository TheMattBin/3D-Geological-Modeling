import pandas as pd
import numpy as np

geodf = pd.read_csv(r'D:\HK Model Arcgis\HK Model ArcPro\Boreline Trial\airport_geo_comb.csv')
sptdf = pd.read_csv(r'D:\HK Model Arcgis\HK Model ArcPro\Boreline Trial\airport.csv')
spt_geo_comb_df = pd.read_csv(r'D:\pythonProject\3D Modelling\SPT Model\sptgeo_comb.csv')
SHB_data = pd.read_csv(r'D:\pythonProject\BH Data Processing\AGS_merge\AGS_merge\Triaxial Test Data V1.csv')

#geodf.set_index(['Rep_BH_ID', 'TopDepth'])
#sptdf.set_index(['Location ID', 'Depth Top'])

#df2 = spt_geo_comb_df.merge(sptdf, left_on='Rep_BH_ID', right_on='Rep_BH_ID')
#df3 = df2[(df2['Depth Top'] <= df2['TopDepth']) & (df2['TopDepth'] < df2['Depth Base'])]
df_comb_w_shearbox = spt_geo_comb_df.merge(SHB_data, left_on='Hole ID', right_on='HOLE_ID')
df_comb_w_shearbox = df_comb_w_shearbox[(df_comb_w_shearbox['Depth Top'] <= df_comb_w_shearbox['SAMP_TOP']) & (df_comb_w_shearbox['SAMP_TOP'] < df_comb_w_shearbox['Depth Base'])]
print(df_comb_w_shearbox)
df_comb_w_shearbox.to_csv('sptgeo_comb_tri.csv')