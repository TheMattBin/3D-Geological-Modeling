import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
import netCDF4 as nc
import os
from netCDF4 import stringtochar
import rasterio
import numpy.ma as ma

var2 = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model for whole HK\kNN_airport_250x250x2_100 distance.nc')
Seabed_Raster = rasterio.open(r'D:\HK Model Arcgis\HK Model ArcPro\Demo_21Sept\HK Model 21 Sept\seabed_no_hk.tif')
shenzhen_Raster = rasterio.open(r'C:\Users\Matthew\Desktop\Suggested Zones\ShenzhenArea_PolygonToRaster1.tif')
var = nc.Dataset(r'D:\pythonProject\3D Modelling\3D Model for whole HK\removeland_noZHv2.nc')


def model_land_no_(var):
    litoMatrix = var['Lithology'][:]
    print(var.dimensions)
    X = var['x'][:]
    Y = var['y'][:]
    Z = var['z'][:]
    north = litoMatrix.shape[1]
    depth = litoMatrix.shape[0]
    east = litoMatrix.shape[2]
    print(litoMatrix[0])
    land = []
    for j in range(north):
        for i in range(east):
            if 800576.1164593603 <= X[i] <= 870476.1164593603 and 834121.0385106392 <= Y[j] <= 858521.0385106392:
                if shenzhen_Raster.index(X[i], Y[j]):
                    land.append([j,i])
            if 801975.0 <= X[i] <= 860025.0 and 800975.0 <= Y[j] <= 847525.0:
                x, y = Seabed_Raster.index(X[i], Y[j])
                val = Seabed_Raster.read(1)[x, y]
                if val <= -3.4e+38:
                    land.append([j,i])
    for l in land:
        litoMatrix[:, l[0], l[1]] =12
    print(litoMatrix)

    # Create NetCDF
    outDataSet = nc.Dataset('removeland.nc', 'w', format = 'NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z',len(Z))
    outDataSet.createDimension('y',len(Y))
    outDataSet.createDimension('x',len(X))

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology = outDataSet.createVariable('Lithology', int, ('z', 'y', 'x'), fill_value = -9999)

    # Assign values
    ncX[:] = X
    ncY[:] = Y
    ncZ[:] = Z
    ncLithology[:,:,:] = litoMatrix

    ncLithology.long_name = 'Lithology'

    ncZ.positive = 'down'

    ncY.standard_name = 'projection_y_coordinate'
    ncY.units = 'm'

    ncX.standard_name = 'projection_x_coordinate'
    ncX.units = 'm'

    var.close()
    outDataSet.close()

def model_land_no_SZ(var):
    litoMatrix = var['Lithology'][:]
    print(var.dimensions)
    X = var['x'][:]
    Y = var['y'][:]
    Z = var['z'][:]
    north = litoMatrix.shape[1]
    depth = litoMatrix.shape[0]
    east = litoMatrix.shape[2]
    print(litoMatrix[0])
    land = []
    out_hk = []
    for j in range(north):
        for i in range(east):
            if 801975.0 <= X[i] <= 860025.0 and 800975.0 <= Y[j] <= 847525.0:
                x, y = Seabed_Raster.index(X[i], Y[j])
                val = Seabed_Raster.read(1)[x, y]
                if val <= -3.4e+38:
                    land.append([j,i])
            else:
                out_hk.append([j,i])
    for l in land:
        litoMatrix[:, l[0], l[1]] =12
    for l in out_hk:
        litoMatrix[:, l[0], l[1]] = 13
    print(litoMatrix)

    # Create NetCDF
    outDataSet = nc.Dataset('removeland_noZHv2.nc', 'w', format = 'NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z',len(Z))
    outDataSet.createDimension('y',len(Y))
    outDataSet.createDimension('x',len(X))

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology = outDataSet.createVariable('Lithology', int, ('z', 'y', 'x'), fill_value = -9999)

    # Assign values
    ncX[:] = X
    ncY[:] = Y
    ncZ[:] = Z
    ncLithology[:,:,:] = litoMatrix

    ncLithology.long_name = 'Lithology'

    ncZ.positive = 'down'

    ncY.standard_name = 'projection_y_coordinate'
    ncY.units = 'm'

    ncX.standard_name = 'projection_x_coordinate'
    ncX.units = 'm'

    outDataSet.esri_pe_string = 'PROJCS["Hong_Kong_1980_Grid",GEOGCS["GCS_Hong_Kong_1980",' \
                                'DATUM["D_Hong_Kong_1980",SPHEROID["International_1924",6378388.0,297.0]],' \
                                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],' \
                                'PARAMETER["False_Easting",836694.05],PARAMETER["False_Northing",819069.8],PARAMETER["Central_Meridian",114.1785555555556],' \
                                'PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",22.31213333333334],UNIT["Meter",1.0]],' \
                                'VERTCS["Hong_Kong_Principal_Datum",VDATUM["Hong_Kong_Principal_Datum"],PARAMETER["Vertical_Shift",0.0],' \
                                'PARAMETER["Direction",1.0],UNIT["Meter",1.0]]'

    var.close()
    outDataSet.close()

def model_hide_land(var):
    litoMatrix = var['Lithology'][:]
    print(var.dimensions)
    X = var['x'][:]
    Y = var['y'][:]
    Z = var['z'][:]
    north = litoMatrix.shape[1]
    depth = litoMatrix.shape[0]
    east = litoMatrix.shape[2]
    print(litoMatrix[0])
    land = []
    out_hk = []
    for k in range(depth):
        for j in range(north):
            for i in range(east):
                if litoMatrix[k, j, i] == 12 or litoMatrix[k, j, i] == 13:
                    litoMatrix[k, j, i] = ma.masked
    print(litoMatrix)

    # Create NetCDF
    outDataSet = nc.Dataset('removeland_hide_all_land.nc', 'w', format = 'NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z',len(Z))
    outDataSet.createDimension('y',len(Y))
    outDataSet.createDimension('x',len(X))

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology = outDataSet.createVariable('Lithology', 'i1', ('z', 'y', 'x'), fill_value = -9999)

    # Assign values
    ncX[:] = X
    ncY[:] = Y
    ncZ[:] = Z
    ncLithology[:,:,:] = litoMatrix

    ncLithology.long_name = 'Lithology'

    ncZ.positive = 'down'

    ncY.standard_name = 'projection_y_coordinate'
    ncY.units = 'm'

    ncX.standard_name = 'projection_x_coordinate'
    ncX.units = 'm'

    outDataSet.esri_pe_string = 'PROJCS["Hong_Kong_1980_Grid",GEOGCS["GCS_Hong_Kong_1980",' \
                                'DATUM["D_Hong_Kong_1980",SPHEROID["International_1924",6378388.0,297.0]],' \
                                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],' \
                                'PARAMETER["False_Easting",836694.05],PARAMETER["False_Northing",819069.8],PARAMETER["Central_Meridian",114.1785555555556],' \
                                'PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",22.31213333333334],UNIT["Meter",1.0]],' \
                                'VERTCS["Hong_Kong_Principal_Datum",VDATUM["Hong_Kong_Principal_Datum"],PARAMETER["Vertical_Shift",0.0],' \
                                'PARAMETER["Direction",1.0],UNIT["Meter",1.0]]'

    var.close()
    outDataSet.close()

model_hide_land(var)