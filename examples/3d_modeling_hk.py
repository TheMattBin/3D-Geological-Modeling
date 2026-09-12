import numpy as np
import pandas as pd
import numpy.ma as ma

import rasterio
import netCDF4

from scipy.stats import entropy
from sklearn import neighbors
from sklearn.metrics import confusion_matrix


# --- Constants and file paths ---
HK_MODEL_PATH = 'HK_BH_model.csv'
SEABED_RASTER_PATH = 'seabed_no_hk.tif'
SHENZHEN_RASTER_PATH = 'ShenzhenArea_PolygonToRaster1.tif'
VOXEL_DIST = 100
N_NEIGHBORS = 15


def voxel_model(df, dist):
    # Build lithology point cloud
    litoPoints = []
    for _, values in df.iterrows():
        wellX, wellY = values.Easting, values.Northing
        wellXY = [wellX, wellY]
        top = values['Ground Level'] - values['Depth Top']
        base = values['Ground Level'] - values['Depth Base']
        litoPoints.append(wellXY + [top, values['Legend Code']])
        litoPoints.append(wellXY + [base, values['Legend Code']])
        litoLength = top - base
        if litoLength < 1:
            # Add midpoint if very thin
            midPoint = wellXY + [top - litoLength / 2, values['Legend Code']]
            litoPoints.append(midPoint)
        else:
            npoints = int(litoLength)
            for point in range(1, npoints + 1):
                z = top - litoLength * point / (npoints + 1)
                litoPoints.append(wellXY + [z, values['Legend Code']])

    litoNp = np.array(litoPoints)
    coor_trans = np.hstack((litoNp[:, :2] / dist, litoNp[:, 2].reshape(-1, 1)))
    soil_class = litoNp[:, 3]

    # Fit kNN classifier
    clf = neighbors.KNeighborsClassifier(N_NEIGHBORS, weights='distance')
    clf.fit(coor_trans, soil_class)

    # Define voxel grid
    x_min, x_max = litoNp[:, 0].min() - 1, litoNp[:, 0].max() + 1
    y_min, y_max = litoNp[:, 1].min() - 1, litoNp[:, 1].max() + 1
    z_min, z_max = litoNp[:, 2].min() - 1, litoNp[:, 2].max() + 1
    vertexCols = np.arange(x_min, x_max, 500)
    vertexRows = np.arange(y_min, y_max, 500)
    vertexLays = np.arange(z_max, z_min, -1)
    cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
    cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
    cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
    nCols, nRows, nLays = cellCols.shape[0], cellRows.shape[0], cellLays.shape[0]

    litoMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
    ProbMatrix = ma.array(np.zeros([nLays, nRows, nCols]))

    # Open rasters once outside the loop
    Seabed_Raster = rasterio.open(SEABED_RASTER_PATH)
    shenzhen_Raster = rasterio.open(SHENZHEN_RASTER_PATH)

    # Batched voxel predictions (one predict / predict_proba call for the
    # whole grid; loop order lay -> row -> col is preserved by the reshape)
    Xq = np.empty((nLays, nRows, nCols, 3))
    Xq[..., 0] = (cellCols / dist)[None, None, :]
    Xq[..., 1] = (cellRows / dist)[None, :, None]
    Xq[..., 2] = cellLays[:, None, None]
    litoMatrix[:, :, :] = clf.predict(Xq.reshape(-1, 3)).reshape(nLays, nRows, nCols)
    ProbMatrix[:, :, :] = entropy(clf.predict_proba(Xq.reshape(-1, 3)), base=2, axis=1).reshape(nLays, nRows, nCols)

    # Seabed / Shenzhen masking: read each raster band once, then evaluate the
    # per-cell rules (same bboxes, same try/except semantics, and the same
    # scalar -3.4e+38 comparisons — raster.nodata reads back as the
    # float32-rounded -3.3999999521443642e+38, which classifies float32-nodata
    # pixels differently) on the (row, col) grid only — they do not depend on
    # the layer.
    sz_band = shenzhen_Raster.read(1)
    sb_band = Seabed_Raster.read(1)

    # in_sz: inside the Shenzhen polygon (valid pixel) — such cells entered
    # `land` in the old loop and were therefore exempt from the seabed rule.
    # sb_val / sb_has: seabed elevation where the bbox holds and the raster
    # lookup succeeded (NaN otherwise; NaN comparisons are False, matching the
    # old skipped-if branches).
    in_sz = np.zeros((nRows, nCols), dtype=bool)
    sb_val = np.full((nRows, nCols), np.nan)
    sb_has = np.zeros((nRows, nCols), dtype=bool)
    sb_nodata_2d = np.zeros((nRows, nCols), dtype=bool)
    for row in range(nRows):
        for col in range(nCols):
            cc, cr = cellCols[col], cellRows[row]
            # Shenzhen mask
            if 800576.1164593603 <= cc <= 870476.1164593603 and 834121.0385106392 <= cr <= 858521.0385106392:
                try:
                    x_sz, y_sz = shenzhen_Raster.index(cc, cr)
                    sz_val = sz_band[x_sz, y_sz]
                    if sz_val > -3.4e+38:
                        in_sz[row, col] = True
                except Exception:
                    pass
            # Seabed mask
            if 801975.0 <= cc <= 860025.0 and 800975.0 <= cr <= 847525.0:
                try:
                    x, y = Seabed_Raster.index(cc, cr)
                    sb_val[row, col] = sb_band[x, y]
                    sb_has[row, col] = True
                    if sb_band[x, y] <= -3.4e+38:
                        sb_nodata_2d[row, col] = True
                except Exception:
                    pass

    # Old per-voxel rule: nodata seabed pixels had their entropy masked at
    # every layer (regardless of land membership), and the water rule
    # (lithology 0 + masked entropy) applied only where the coordinate was
    # NOT in land — land being exactly the Shenzhen-valid and seabed-nodata
    # cells, already blocked at the first layer.
    ProbMatrix[np.broadcast_to(sb_nodata_2d[None, :, :], ProbMatrix.shape)] = ma.masked
    water = sb_has & ~sb_nodata_2d & ~in_sz & (cellLays[:, None, None] >= sb_val[None, :, :])
    litoMatrix[water] = 0
    ProbMatrix[water] = ma.masked

    # Confusion matrix (for reporting, not used further)
    predicted = clf.predict(coor_trans)
    _ = confusion_matrix(soil_class, predicted)

    return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix


def createVoxel(cellCols, cellRows, cellLays, litoMatrix, ProbMatrix, dist):
    nCols, nRows, nLays = cellCols.shape[0], cellRows.shape[0], cellLays.shape[0]
    outDataSet = netCDF4.Dataset(f'kNN_airport_250x250x2_{dist} distance.nc', 'w', format='NETCDF4')
    # Create dimensions
    outDataSet.createDimension('z', nLays)
    outDataSet.createDimension('y', nRows)
    outDataSet.createDimension('x', nCols)
    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))
    ncLithology_m = outDataSet.createVariable('Lithology', 'i1', ('z', 'y', 'x'), fill_value=-9999)
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
    df_hk = pd.read_csv(HK_MODEL_PATH)
    a, b, c, d, e = voxel_model(df_hk, VOXEL_DIST)
    createVoxel(a, b, c, d, e, VOXEL_DIST)


if __name__ == "__main__":
    main()