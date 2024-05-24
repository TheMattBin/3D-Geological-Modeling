"""
================================
3D Geotechnical Model using Nearest Neighbors Classification
================================

It will plot the decision boundaries for each soil/lithology class.

Author: Matthew Y.B. Liu, 29th Dec. 2021

"""

import numpy as np
import pandas as pd
from sklearn import neighbors
import netCDF4
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn import svm
import rasterio
from rasterio.plot import show
import numpy.ma as ma


# Read csv file
sptdf_airport = pd.read_csv(r'D:\HK Model Arcgis\HK Model ArcPro\Boreline Trial\airport.csv')


def GeoTech_model(df, dist):
    # point cloud of spt-n
    SPTPoints = []

    for index, values in df.iterrows():
        wellX, wellY = values.Easting, values.Northing
        wellXY = [wellX, wellY]
        SPTPoints.append(wellXY + [values['MidElev'], values['SPTN']])

    SPTNNp = np.array(SPTPoints)
    coor_trans = np.hstack((SPTNNp[:, :2] / dist, SPTNNp[:, 2].reshape(-1, 1)))
    spt_class = SPTNNp[:, 3]

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(15, weights='distance')
    clf.fit(coor_trans, spt_class)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    # need to match with geological model
    x_min, x_max = SPTNNp[:, 0].min() - 1, SPTNNp[:, 0].max() + 1
    y_min, y_max = SPTNNp[:, 1].min() - 1, SPTNNp[:, 1].max() + 1
    z_min, z_max = SPTNNp[:, 2].min() - 1, SPTNNp[:, 2].max() + 1

    # Determination of the lithology matrix
    vertexCols = np.arange(x_min, x_max, 250)
    vertexRows = np.arange(y_min, y_max, 250)
    vertexLays = np.arange(z_max, z_min, -2)
    cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
    cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
    cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    litoMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
    ProbMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
    land = []
    for lay in tqdm(range(nLays)):
        for row in range(nRows):
            for col in range(nCols):
                # predict the soil class of unknown points and the info entropy
                cellXYZ = [cellCols[col], cellRows[row], cellLays[lay]]
                cellTrans = np.array([cellCols[col] / dist, cellRows[row] / dist, cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)

    return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix


def createVoxel(cellCols, cellRows, cellLays, litoMatrix, ProbMatrix, dist):
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    # Create NetCDF
    outDataSet = netCDF4.Dataset('kNN_airport_250x250x2_GeoTech{} distance.nc'.format(dist), 'w', format = 'NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z',nLays)
    outDataSet.createDimension('y',nRows)
    outDataSet.createDimension('x',nCols)

    ncZ = outDataSet.createVariable('z', 'f4', ('z'))
    ncY = outDataSet.createVariable('y', 'f4', ('y'))
    ncX = outDataSet.createVariable('x', 'f4', ('x'))

    ncLithology_m = outDataSet.createVariable('Lithology', 'f4', ('z', 'y', 'x'), fill_value = -9999)
    info_entro_m = outDataSet.createVariable('Information Entropy', 'f4', ('z', 'y', 'x'), fill_value = -9999)

    # Assign values
    ncX[:] = cellCols
    ncY[:] = cellRows
    ncZ[:] = cellLays

    ncLithology_m[:,:,:] = litoMatrix
    info_entro_m[:,:,:] = ProbMatrix

    ncLithology_m.long_name = 'Lithology'
    info_entro_m.long_name = 'Entropy'

    ncZ.positive = 'up'

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
for i in [100]:
    a,b,c,d,e = GeoTech_model(sptdf_airport, i)
    createVoxel(a,b,c,d,e, i)