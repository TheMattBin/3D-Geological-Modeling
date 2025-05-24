import pandas as pd
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from datetime import datetime

import rasterio
import netCDF4
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingClassifier


def point_cloud_trans(df, dist):
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
    coor_trans = np.hstack((litoNp[:, :2] / dist, litoNp[:, 2].reshape(-1, 1)))
    soil_class = litoNp[:, 3]
    return litoNp, coor_trans, soil_class


def voxel_model_GBC(coor, coor_trans, soil_class, ne, dist, seabed_raster):
    """
    Build a 3D Gradient Boosting geotechnical model and return grid and result matrices.
    """
    clf = GradientBoostingClassifier(n_estimators=ne)
    clf.fit(coor_trans, soil_class)
    x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
    y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
    z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1
    vertexCols = np.arange(x_min, x_max, 50)
    vertexRows = np.arange(y_min, y_max, 50)
    vertexLays = np.arange(z_max, z_min, -1)
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
                cellTrans = np.array([cellCols[col] / dist, cellRows[row] / dist, cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)
                x, y = seabed_raster.index(cellCols[col], cellRows[row])
                val = seabed_raster.read(1)[x, y]
                if val <= -3.4e+38:
                    ProbMatrix[lay, row, col] = ma.masked
                    land.append([cellCols[col], cellRows[row]])
                if cellLays[lay] >= val and [cellCols[col], cellRows[row]] not in land:
                    litoMatrix[lay, row, col] = 0
                    ProbMatrix[lay, row, col] = ma.masked
    return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix


def createVoxel(cellCols, cellRows, cellLays, litoMatrix, ProbMatrix, filename):
    """
    Create a NetCDF file for the voxel data.
    """
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
    outDataSet.esri_pe_string = (
        'PROJCS["Hong_Kong_1980_Grid",GEOGCS["GCS_Hong_Kong_1980",'
        'DATUM["D_Hong_Kong_1980",SPHEROID["International_1924",6378388.0,297.0]],'
        'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
        'PARAMETER["False_Easting",836694.05],PARAMETER["False_Northing",819069.8],PARAMETER["Central_Meridian",114.1785555555556],'
        'PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",22.31213333333334],UNIT["Meter",1.0]],'
        'VERTCS["Hong_Kong_Principal_Datum",VDATUM["Hong_Kong_Principal_Datum"],PARAMETER["Vertical_Shift",0.0],'
        'PARAMETER["Direction",1.0],UNIT["Meter",1.0]]'
    )
    outDataSet.close()


def main():
    # Parameters
    csv_path = 'DCentralWater_20211230.csv'
    seabed_raster_path = 'seabed_no_hk.tif'
    dist = 100
    n_estimators = 400
    output_filename = f'myOutputGBC-pengchau{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.nc'

    # Read data
    df = pd.read_csv(csv_path)
    seabed_raster = rasterio.open(seabed_raster_path)

    # Transform and model
    litoNp, coor_trans, soil_class = point_cloud_trans(df, dist)
    a, b, c, d, e = voxel_model_GBC(litoNp, coor_trans, soil_class, n_estimators, dist, seabed_raster)

    # Create output file
    createVoxel(a, b, c, d, e, output_filename)


if __name__ == "__main__":
    main()