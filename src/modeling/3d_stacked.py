import numpy as np
import pandas as pd
from tqdm import tqdm
import netCDF4

from scipy.stats import entropy
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

def voxel_model_Stacked(df):
    """
    Build a 3D stacked classifier geotechnical model and return grid and result matrices.
    """
    coor = np.stack((df['Easting'].values, df['Northing'].values, df['ElevMid'].values), axis=-1)
    coor_trans = np.hstack((coor[:, :2] / 1000, df['ElevMid'].values.reshape(len(df),1)))
    soil_class = df['Legend Code'].values
    estimators = [
        ('knn', neighbors.KNeighborsClassifier(15, weights='distance')),
        ('rf', RandomForestClassifier(80))
    ]
    clf = StackingClassifier(estimators=estimators)
    clf.fit(coor_trans, soil_class)
    x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
    y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
    z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1
    vertexCols = np.arange(x_min, x_max, 1000)
    vertexRows = np.arange(y_min, y_max, 1000)
    vertexLays = np.arange(z_max, z_min, -1)
    cellCols = (vertexCols[1:]+vertexCols[:-1])/2
    cellRows = (vertexRows[1:]+vertexRows[:-1])/2
    cellLays = (vertexLays[1:]+vertexLays[:-1])/2
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]
    litoMatrix = np.zeros([nLays, nRows, nCols])
    ProbMatrix = np.zeros([nLays, nRows, nCols])
    for lay in tqdm(range(nLays)):
        for row in range(nRows):
            for col in range(nCols):
                cellTrans = np.array([cellCols[col]/1000, cellRows[row]/1000, cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)
    # confusion matrix (not used)
    return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix

def createVoxel(cellCols, cellRows, cellLays, litoMatrix, ProbMatrix, filename):
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
    ncLithology_m = outDataSet.createVariable('Lithology_m', int, ('z', 'y', 'x'), fill_value=-9999)
    info_entro_m = outDataSet.createVariable('info entropy_m', np.float32, ('z', 'y', 'x'), fill_value=-9999)
    ncX[:] = cellCols
    ncY[:] = cellRows
    ncZ[:] = cellLays
    ncLithology_m[:, :, :] = litoMatrix
    info_entro_m[:, :, :] = ProbMatrix
    ncLithology_m.long_name = 'Lithology'
    info_entro_m.long_name = 'Entropy'
    ncZ.positive = 'down'
    ncY.standard_name = 'projection_y_coordinate'
    ncY.units = 'm'
    ncX.standard_name = 'projection_x_coordinate'
    ncX.units = 'm'
    outDataSet.close()

def main():
    # Parameters
    csv_path = 'CentralWater_20211230.csv'
    output_filename = 'myOutputStack1000.nc'
    df = pd.read_csv(csv_path)
    a, b, c, d, e = voxel_model_Stacked(df)
    createVoxel(a, b, c, d, e, output_filename)

if __name__ == "__main__":
    main()
