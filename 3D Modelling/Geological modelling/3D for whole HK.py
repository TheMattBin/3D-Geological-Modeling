from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import netCDF4
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from datetime import datetime
import rasterio
from rasterio.plot import show
import numpy.ma as ma
from sklearn import neighbors

# read file
#df = pd.read_csv(r'D:\pythonProject\BH Data Processing\CW_New_20211220.csv')
df = pd.read_csv(r'D:\pythonProject\BH Data Processing\CentralWater_20211230_comb.csv')
df_pc = pd.read_csv(r'D:\pythonProject\BH Data Processing\CentralWater_20211230_pengchau.csv')
#df = pd.read_csv(r'D:\pythonProject\BH Data Processing\test5.csv')
df_airport = pd.read_csv(r'D:\HK Model Arcgis\HK Model ArcPro\Boreline Trial\airport_geo_comb.csv')
df_hk = pd.read_csv(r'D:\pythonProject\BH Data Processing\HK_BH_model.csv')
Seabed_Raster = rasterio.open(r'D:\HK Model Arcgis\HK Model ArcPro\Demo_21Sept\HK Model 21 Sept\seabed_no_hk.tif')
shenzhen_Raster = rasterio.open(r'C:\Users\Matthew\Desktop\Suggested Zones\ShenzhenArea_PolygonToRaster1.tif')

print(Seabed_Raster.bounds)

def voxel_model(df, dist):
    # neighbours defined
    n_neighbors = 15

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
    coor_trans = np.hstack((litoNp[:, :2] / dist, litoNp[:, 2].reshape(-1, 1)))

    # add customized boundary
    # (df[(df['Easting'] >= 820000) & (df['Easting'] <= 830000)])
    soil_class = litoNp[:,3]

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(coor_trans, soil_class)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = litoNp[:, 0].min() - 1, litoNp[:, 0].max() + 1
    y_min, y_max = litoNp[:, 1].min() - 1, litoNp[:, 1].max() + 1
    z_min, z_max = litoNp[:, 2].min() - 1, litoNp[:, 2].max() + 1

    #Determination of the lithology matrix
    vertexCols = np.arange(x_min, x_max, 500)
    vertexRows = np.arange(y_min, y_max, 500)
    vertexLays = np.arange(z_max, z_min, -1)
    cellCols = (vertexCols[1:]+vertexCols[:-1])/2
    cellRows = (vertexRows[1:]+vertexRows[:-1])/2
    cellLays = (vertexLays[1:]+vertexLays[:-1])/2
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
                cellXYZ = [cellCols[col],cellRows[row],cellLays[lay]]
                cellTrans = np.array([cellCols[col]/dist, cellRows[row]/dist, cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)
                if 800576.1164593603 <= cellCols[col] <= 870476.1164593603 and 834121.0385106392 <= cellRows[row] <= 858521.0385106392:
                    if shenzhen_Raster.index(cellCols[col], cellRows[row]):
                        land.append([cellCols[col], cellRows[row]])
                if 801975.0 <= cellCols[col] <= 860025.0 and 800975.0 <= cellRows[row] <= 847525.0:
                    x,y = Seabed_Raster.index(cellCols[col], cellRows[row])
                    val = Seabed_Raster.read(1)[x, y]
                    if val <= -3.4e+38:
                        ProbMatrix[lay, row, col] = ma.masked
                        land.append([cellCols[col], cellRows[row]])
                    if cellLays[lay] >= val and [cellCols[col], cellRows[row]] not in land:
                        litoMatrix[lay, row, col] = 0
                        ProbMatrix[lay, row, col] = ma.masked


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

def createVoxel(cellCols, cellRows, cellLays, litoMatrix, ProbMatrix, dist):
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    # Create NetCDF
    outDataSet = netCDF4.Dataset('kNN_airport_250x250x2_{} distance.nc'.format(dist), 'w', format = 'NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z',nLays)
    outDataSet.createDimension('y',nRows)
    outDataSet.createDimension('x',nCols)

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology_m = outDataSet.createVariable('Lithology', 'i1', ('z', 'y', 'x'), fill_value = -9999)
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

    outDataSet.esri_pe_string = 'PROJCS["Hong_Kong_1980_Grid",GEOGCS["GCS_Hong_Kong_1980",' \
                                'DATUM["D_Hong_Kong_1980",SPHEROID["International_1924",6378388.0,297.0]],' \
                                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],' \
                                'PARAMETER["False_Easting",836694.05],PARAMETER["False_Northing",819069.8],PARAMETER["Central_Meridian",114.1785555555556],' \
                                'PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",22.31213333333334],UNIT["Meter",1.0]],' \
                                'VERTCS["Hong_Kong_Principal_Datum",VDATUM["Hong_Kong_Principal_Datum"],PARAMETER["Vertical_Shift",0.0],' \
                                'PARAMETER["Direction",1.0],UNIT["Meter",1.0]]'

a,b,c,d,e = voxel_model(df_hk, 100)
createVoxel(a,b,c,d,e,100)