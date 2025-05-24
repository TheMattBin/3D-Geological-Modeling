from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
import netCDF4
from scipy.stats import entropy
from sklearn.neural_network import MLPClassifier

def point_cloud_trans(df):
    """
    Transform dataframe to point cloud and coordinates for modeling.
    """
    litoPoints = []
    for _, values in df.iterrows():
        wellX, wellY = values.Easting, values.Northing
        wellXY = [wellX, wellY]
        litoPoints.append(wellXY + [values['Ground Level'] - values['Depth Top'], values['Legend Code']])
        litoPoints.append(wellXY + [values['Ground Level'] - values['Depth Base'], values['Legend Code']])
        litoLength = (values['Ground Level'] - values['Depth Top']) - (values['Ground Level'] - values['Depth Base'])
        if litoLength < 1:
            pass
        else:
            npoints = int(litoLength)
            for point in range(1, npoints + 1):
                disPoint = wellXY + [
                    (values['Ground Level'] - values['Depth Top']) - litoLength * point / (npoints + 1),
                    values['Legend Code']]
                litoPoints.append(disPoint)
    litoNp = np.array(litoPoints)
    coor_trans = np.hstack((litoNp[:, :2] / 100, litoNp[:, 2].reshape(-1, 1)))
    soil_class = litoNp[:, 3]
    return litoNp, coor_trans, soil_class

def voxel_model_NN(coor, coor_trans, soil_class, alpha):
    """
    Build a 3D neural network geotechnical model and return grid and result matrices.
    """
    clf = MLPClassifier(activation='logistic', solver='adam', hidden_layer_sizes=(25, 25, 25), max_iter=10000, alpha=alpha)
    clf.fit(coor_trans, soil_class)
    x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
    y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
    z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1
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
                cellTrans = np.array([cellCols[col] / 100, cellRows[row] / 100, cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)
    return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix

def createVoxel(cellCols, cellRows, cellLays, litoMatrix, ProbMatrix, filename):
    t = datetime.now()
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]
    outDataSet = netCDF4.Dataset(filename, 'w', format='NETCDF4')
    outDataSet.createDimension('z', nLays)
    outDataSet.createDimension('y', nRows)
    outDataSet.createDimension('x', nCols)
    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))
    ncLithology_m = outDataSet.createVariable('Lithology', int, ('z', 'y', 'x'), fill_value=-9999)
    info_entro_m = outDataSet.createVariable('Information Entropy', np.float32, ('z', 'y', 'x'), fill_value=-9999)
    ncX[:] = cellCols
    ncY[:] = cellRows
    ncZ[:] = cellLays
    ncLithology_m[:, :, :] = litoMatrix
    info_entro_m[:, :, :] = ProbMatrix
    ncLithology_m.long_name = 'Lithology'
    info_entro_m.long_name = 'Entropy'
    ncZ.positive = 'up'
    ncY.standard_name = 'projection_y_coordinate'
    ncY.units = 'm'
    ncX.standard_name = 'projection_x_coordinate'
    ncX.units = 'm'
    outDataSet.close()

def main():
    # Parameters
    csv_path = 'CentralWater_20211230.csv'
    df = pd.read_csv(csv_path)
    litoNp, coor_trans, soil_class = point_cloud_trans(df)
    for i in np.arange(0.01, 0.1, 0.01):
        print(i)
        a, b, c, d, e = voxel_model_NN(litoNp, coor_trans, soil_class, i)
        filename = f'myOutputRF1000-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}-alpha{i:.2f}.nc'
        createVoxel(a, b, c, d, e, filename)

if __name__ == "__main__":
    main()