from pykrige.ok3d import OrdinaryKriging3D
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import netCDF4
from datetime import datetime

df = pd.read_csv(r'D:\pythonProject\BH Data Processing\SPT_CW.csv')

print(df.head())

# add customized boundary
# (df[(df['Easting'] >= 820000) & (df['Easting'] <= 830000)])
coor = np.stack((df['Easting'].values, df['Northing'].values, df['MidElev'].values), axis=-1)
#coor_trans = np.hstack((coor[:, :2], df['ElevMid'].values.reshape(len(df), 1)))
SPTN = df['SPTN'].values

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1

# Determination of the lithology matrix
vertexCols = np.arange(x_min, x_max, 1000)
vertexRows = np.arange(y_min, y_max, 1000)
vertexLays = np.arange(z_max, z_min, -1)

ok3d = OrdinaryKriging3D(
    coor[:, 0], coor[:, 1], coor[:, 2], SPTN, variogram_model="exponential"
)
k3d1, ss3d = ok3d.execute("grid", vertexCols, vertexRows, vertexLays)

print(k3d1.shape)

nCols = vertexCols.shape[0]
nRows = vertexRows.shape[0]
nLays = vertexLays.shape[0]

t = datetime.now()

# Create NetCDF
outDataSet = netCDF4.Dataset('myOutputGBC-{}.nc'.format(t.strftime('%d-%m-%Y-%H-%M-%S')), 'w', format='NETCDF4')

# Create dimensions
outDataSet.createDimension('z', nLays)
outDataSet.createDimension('y', nRows)
outDataSet.createDimension('x', nCols)

ncZ = outDataSet.createVariable('z', np.float32, ('z'))
ncY = outDataSet.createVariable('y', np.float32, ('y'))
ncX = outDataSet.createVariable('x', np.float32, ('x'))

info_entro_m = outDataSet.createVariable('SPT-N', np.float32, ('z', 'y', 'x'), fill_value=-9999)

# Assign values
ncX[:] = vertexCols
ncY[:] = vertexRows
ncZ[:] = vertexLays


info_entro_m[:, :, :] = k3d1


info_entro_m.long_name = 'SPT-N'

ncZ.positive = 'up'

ncY.standard_name = 'projection_y_coordinate'
ncY.units = 'm'

ncX.standard_name = 'projection_x_coordinate'
ncX.units = 'm'