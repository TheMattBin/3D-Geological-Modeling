from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
import numpy as np
import pandas as pd
from sklearn import svm
import netCDF4
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from datetime import datetime
from sklearn import neighbors

# read file
#df = pd.read_csv(r'D:\pythonProject\BH Data Processing\CW_New_20211220.csv')
df = pd.read_csv(r'D:\pythonProject\BH Data Processing\CentralWater_20211230.csv')
df2 = pd.read_csv(r'D:\pythonProject\3D Modelling\3D Model SVM\AirPort.csv')

def voxel_model_Stacked(df):

    # add customized boundary
    # (df[(df['Easting'] >= 820000) & (df['Easting'] <= 830000)])
    coor = np.stack((df['Easting'].values, df['Northing'].values, df['ElevMid'].values), axis=-1)
    coor_trans = np.hstack((coor[:, :2] / 1000, df['ElevMid'].values.reshape(len(df),1)))
    soil_class = df['Legend Code'].values


    # we create an instance of Neighbours Classifier and fit the data.
    estimators = [('knn', neighbors.KNeighborsClassifier(15, weights='distance')), ('rf', RandomForestClassifier(80))]
    clf = StackingClassifier(estimators = estimators)
    clf.fit(coor_trans, soil_class)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
    y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
    z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1

    #Determination of the lithology matrix
    vertexCols = np.arange(x_min, x_max, 1000)
    vertexRows = np.arange(y_min, y_max, 1000)
    vertexLays = np.arange(z_max, z_min, -1)
    cellCols = (vertexCols[1:]+vertexCols[:-1])/2
    cellRows = (vertexRows[1:]+vertexRows[:-1])/2
    cellLays = (vertexLays[1:]+vertexLays[:-1])/2
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    litoMatrix = np.zeros([nLays,nRows,nCols])
    ProbMatrix = np.zeros([nLays,nRows,nCols])
    for lay in tqdm(range(nLays)):
        for row in range(nRows):
            for col in range(nCols):
                # predict the soil class of unknown points and the info entropy
                cellXYZ = [cellCols[col],cellRows[row],cellLays[lay]]
                cellTrans = np.array([cellCols[col]/1000, cellRows[row]/1000, cellLays[lay]])
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
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    # Create NetCDF
    outDataSet = netCDF4.Dataset('myOutputStack1000.nc', 'w', format = 'NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z',nLays)
    outDataSet.createDimension('y',nRows)
    outDataSet.createDimension('x',nCols)

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology_m = outDataSet.createVariable('Lithology_m', int, ('z', 'y', 'x'), fill_value = -9999)
    info_entro_m = outDataSet.createVariable('info entropy_m', np.float32, ('z', 'y', 'x'), fill_value = -9999)

    # Assign values
    ncX[:] = cellCols
    ncY[:] = cellRows
    ncZ[:] = cellLays

    ncLithology_m[:,:,:] = litoMatrix
    info_entro_m[:,:,:] = ProbMatrix

    ncLithology_m.long_name = 'Lithology'
    info_entro_m.long_name = 'Entropy'

    ncZ.positive = 'down'

    ncY.standard_name = 'projection_y_coordinate'
    ncY.units = 'm'

    ncX.standard_name = 'projection_x_coordinate'
    ncX.units = 'm'

a,b,c,d,e = voxel_model_Stacked(df)
createVoxel(a,b,c,d,e)
