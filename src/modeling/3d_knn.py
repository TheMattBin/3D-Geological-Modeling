"""
3D Geological Model using Nearest Neighbors Classification

Author: Matthew Y.B. Liu, 29th Dec. 2021
"""

import pandas as pd
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import netCDF4

from sklearn import neighbors
from scipy.stats import entropy

# --- Parameters ---
CSV_PATH = 'airport.csv'
DIST = 100
OUTPUT_FILENAME = f'kNN_airport_250x250x2_GeoTech{DIST}_distance.nc'

def voxel_model(
    df: pd.DataFrame,
    dist: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 3D KNN geotechnical model and return grid and result matrices.
    """
    n_neighbors = 15
    litoPoints = []
    for _, values in df.iterrows():
        wellX, wellY = values.Easting, values.Northing
        litoPoints.append([wellX, wellY, values['MidElev'], values['SPTN']])
    SPTNNp = np.array(litoPoints)
    coor_trans = np.hstack((SPTNNp[:, :2] / dist, SPTNNp[:, 2].reshape(-1, 1)))
    spt_class = SPTNNp[:, 3]
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(coor_trans, spt_class)
    x_min, x_max = SPTNNp[:, 0].min() - 1, SPTNNp[:, 0].max() + 1
    y_min, y_max = SPTNNp[:, 1].min() - 1, SPTNNp[:, 1].max() + 1
    z_min, z_max = SPTNNp[:, 2].min() - 1, SPTNNp[:, 2].max() + 1
    vertexCols = np.arange(x_min, x_max, 250)
    vertexRows = np.arange(y_min, y_max, 250)
    vertexLays = np.arange(z_max, z_min, -2)
    cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
    cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
    cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
    nCols, nRows, nLays = cellCols.shape[0], cellRows.shape[0], cellLays.shape[0]
    litoMatrix = ma.zeros([nLays, nRows, nCols])
    ProbMatrix = ma.zeros([nLays, nRows, nCols])
    for lay in tqdm(range(nLays), desc="Processing layers"):
        for row in range(nRows):
            for col in range(nCols):
                cellTrans = np.array([cellCols[col] / dist, cellRows[row] / dist, cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)
    return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix

def create_voxel(
    cellCols: np.ndarray,
    cellRows: np.ndarray,
    cellLays: np.ndarray,
    litoMatrix: np.ndarray,
    ProbMatrix: np.ndarray,
    filename: str
) -> None:
    """
    Write the voxel model and entropy to a NetCDF file.
    """
    nCols, nRows, nLays = cellCols.shape[0], cellRows.shape[0], cellLays.shape[0]
    with netCDF4.Dataset(filename, 'w', format='NETCDF4') as outDataSet:
        outDataSet.createDimension('z', nLays)
        outDataSet.createDimension('y', nRows)
        outDataSet.createDimension('x', nCols)
        ncZ = outDataSet.createVariable('z', 'f4', ('z',))
        ncY = outDataSet.createVariable('y', 'f4', ('y',))
        ncX = outDataSet.createVariable('x', 'f4', ('x',))
        ncLithology_m = outDataSet.createVariable('Lithology', 'f4', ('z', 'y', 'x'), fill_value=-9999)
        info_entro_m = outDataSet.createVariable('Information Entropy', 'f4', ('z', 'y', 'x'), fill_value=-9999)
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
        outDataSet.esri_pe_string = (
            'PROJCS["Hong_Kong_1980_Grid",GEOGCS["GCS_Hong_Kong_1980",'
            'DATUM["D_Hong_Kong_1980",SPHEROID["International_1924",6378388.0,297.0]],'
            'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
            'PARAMETER["False_Easting",836694.05],PARAMETER["False_Northing",819069.8],PARAMETER["Central_Meridian",114.1785555555556],'
            'PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",22.31213333333334],UNIT["Meter",1.0]],'
            'VERTCS["Hong_Kong_Principal_Datum",VDATUM["Hong_Kong_Principal_Datum"],PARAMETER["Vertical_Shift",0.0],'
            'PARAMETER["Direction",1.0],UNIT["Meter",1.0]]'
        )

def main():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    a, b, c, d, e = voxel_model(df, DIST)
    create_voxel(a, b, c, d, e, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()