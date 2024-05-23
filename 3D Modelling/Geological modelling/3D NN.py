import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import netCDF4
import numpy.ma as ma
from scipy.stats import entropy
from tqdm import tqdm
from datetime import datetime

df = pd.read_csv(r'D:\pythonProject\BH Data Processing\CentralWater_20211230.csv')

def point_cloud_trans(df):

    # Point cloud of lithologies
    litoPoints = []

    for index, values in df.iterrows():
        wellX, wellY = values.Easting, values.Northing
        wellXY = [wellX, wellY]
        litoPoints.append(wellXY + [values['Ground Level'] - values['Depth Top'], values['Legend Code']])
        litoPoints.append(wellXY + [values['Ground Level'] - values['Depth Base'], values['Legend Code']])

        litoLength = (values['Ground Level'] - values['Depth Top']) - (values['Ground Level'] - values['Depth Base'])
        if litoLength < 1:
            midPoint = wellXY + [(values['Ground Level'] - values['Depth Top']) - litoLength / 2, values['Legend Code']]
        else:
            npoints = int(litoLength)
            for point in range(1, npoints + 1):
                disPoint = wellXY + [
                    (values['Ground Level'] - values['Depth Top']) - litoLength * point / (npoints + 1),
                    values['Legend Code']]
                litoPoints.append(disPoint)

    litoNp = np.array(litoPoints)
    coor_trans = np.hstack((litoNp[:, :2] / 100, litoNp[:, 2].reshape(-1, 1)))
    soil_class = litoNp[:,3]

    return litoNp, coor_trans, soil_class

def voxel_model_NN(coor, coor_trans, soil_class, alpha):

    # we create an instance of Neighbours Classifier and fit the data.
    clf = MLPClassifier(activation='logistic',solver='adam',hidden_layer_sizes=(25, 25, 25), max_iter=10000, alpha=alpha)
    clf.fit(coor_trans, soil_class)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
    y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
    z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1

    # Determination of the lithology matrix
    vertexCols = np.arange(x_min, x_max, 1000)
    vertexRows = np.arange(y_min, y_max, 1000)
    vertexLays = np.arange(z_max, z_min, -1)
    cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
    cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
    cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    litoMatrix = np.zeros([nLays, nRows, nCols])
    ProbMatrix = np.zeros([nLays, nRows, nCols])
    for lay in tqdm(range(nLays)):
        for row in range(nRows):
            for col in range(nCols):
                # predict the soil class of unknown points and the info entropy
                cellXYZ = [cellCols[col], cellRows[row], cellLays[lay]]
                cellTrans = np.array([cellCols[col] / 100, cellRows[row] / 100, cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)

    # Determination of confusion matrix
    numberSamples = coor_trans.shape[0]

    predicted = []

    for i in range(numberSamples):
        predicted.append(clf.predict([coor_trans[i]]))
    results = confusion_matrix(soil_class, predicted)

    print(results)

    from sklearn.metrics import classification_report
    print(classification_report(soil_class, predicted))

    return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix

def createVoxel(cellCols, cellRows, cellLays, litoMatrix, ProbMatrix):
    t = datetime.now()

    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    # Create NetCDF
    outDataSet = netCDF4.Dataset('myOutputRF1000-{}.nc'.format(t.strftime('%d-%m-%Y-%H-%M-%S')), 'w', format = 'NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z',nLays)
    outDataSet.createDimension('y',nRows)
    outDataSet.createDimension('x',nCols)

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology_m = outDataSet.createVariable('Lithology', int, ('z', 'y', 'x'), fill_value = -9999)
    info_entro_m = outDataSet.createVariable('Information Entropy', np.float32, ('z', 'y', 'x'), fill_value = -9999)

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

litoNp, coor_trans, soil_class = point_cloud_trans(df)
for i in np.arange(0.01,0.1,0.01):
    print(i)
    a,b,c,d,e = voxel_model_NN(litoNp,coor_trans,soil_class,i)
    createVoxel(a,b,c,d,e)