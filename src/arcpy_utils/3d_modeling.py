"""
================================
3D Geological Model using Nearest Neighbors Classification
================================

It will plot the decision boundaries for each soil/lithology class.

Author: Matthew Y.B. Liu, 29th Dec. 2021

"""

import numpy as np
import pandas as pd
from sklearn import neighbors
import netCDF4
import os
import arcpy
import numpy.ma as ma
from scipy.stats import entropy
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

def knn3d(table, xmin, xmax, ymin, ymax, xresolution, yresolution, zresolution, ratio, seabed, file, name, n_neighbors):

    # read file
    df = pd.read_csv(table)

    # add customized boundary
    df = df[(df['Easting'] >= float(xmin)) & (df['Easting'] <= float(xmax)) & (df['Northing'] >= float(ymin)) & (df['Northing'] <= float(ymax))]

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
    coor_trans = np.hstack((litoNp[:, :2] / float(ratio), litoNp[:, 2].reshape(-1, 1)))
    soil_class = litoNp[:, 3]


    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(int(n_neighbors), weights='distance')
    clf.fit(coor_trans, soil_class)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = litoNp[:, 0].min() - 1, litoNp[:, 0].max() + 1
    y_min, y_max = litoNp[:, 1].min() - 1, litoNp[:, 1].max() + 1
    z_min, z_max = litoNp[:, 2].min() - 1, litoNp[:, 2].max() + 1

    #Determination of the lithology matrix
    zresolution = - float(zresolution)
    vertexCols = np.arange(x_min, x_max, float(xresolution))
    vertexRows = np.arange(y_min, y_max, float(yresolution))
    vertexLays = np.arange(z_max, z_min, zresolution)
    cellCols = (vertexCols[1:]+vertexCols[:-1])/2
    cellRows = (vertexRows[1:]+vertexRows[:-1])/2
    cellLays = (vertexLays[1:]+vertexLays[:-1])/2
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    workspace = arcpy.env.workspace
    arcpy.CreateFeatureclass_management(workspace, 'Pts', "POINT", '', 'DISABLED', 'ENABLED', arcpy.SpatialReference(2326, 5738))
    cur_borept = arcpy.da.InsertCursor('Pts', "SHAPE@")

    litoMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
    ProbMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
    for lay in range(nLays):
        for row in range(nRows):
            for col in range(nCols):
                # predict the soil class of unknown points and the info entropy
                cellXYZ = [cellCols[col], cellRows[row], cellLays[lay]]
                cellTrans = np.array([cellCols[col] / float(ratio), cellRows[row] / float(ratio), cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)
                pt = arcpy.Point(cellCols[col], cellRows[row], cellLays[lay])
                cur_borept.insertRow([pt])

    arcpy.sa.ExtractValuesToPoints("Pts", seabed, "pt_on_seabed", "None", "VALUE_ONLY")

    # store value of each points
    value_seabed = []
    pt_seabed = arcpy.da.SearchCursor("pt_on_seabed", ['SHAPE@XY', 'SHAPE@Z', 'RASTERVALU'])
    for row in pt_seabed:
        if row[2] is not None and row[1] >= row[2]:
            value_seabed.append([format(row[0][0], '.1f'), format(row[0][1], '.1f'), format(row[1], '.1f')])


    flag = False
    # add water layer
    for lay in range(nLays):
        for row in range(nRows):
            for col in range(nCols):
                cellXYZ = [format(cellCols[col], '.1f'), format(cellRows[row], '.1f'), format(cellLays[lay], '.1f')]
                if cellXYZ in value_seabed:
                    litoMatrix[lay, row, col] = 0
                    ProbMatrix[lay, row, col] = ma.masked

    del cur_borept
    del pt_seabed
    arcpy.management.Delete("Pts")
    arcpy.management.Delete("Pts")

    #del point on seabed
    arcpy.management.Delete("pt_on_seabed")
    arcpy.management.Delete("pt_on_seabed")

    # Create NetCDF
    name = name+ '.nc'
    outDataSet = netCDF4.Dataset(os.path.join(file, name), 'w', format = 'NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z',nLays)
    outDataSet.createDimension('y',nRows)
    outDataSet.createDimension('x',nCols)

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology = outDataSet.createVariable('Lithology', 'i1', ('z', 'y', 'x'), fill_value = -9999)
    info_entro = outDataSet.createVariable('Information Entropy', np.float32, ('z', 'y', 'x'), fill_value=-9999)

    # Assign values
    ncX[:] = cellCols
    ncY[:] = cellRows
    ncZ[:] = cellLays

    ncLithology[:, :, :] = litoMatrix
    info_entro[:, :, :] = ProbMatrix

    ncLithology.long_name = 'Lithology'
    info_entro.long_name = 'Entropy'

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

    outDataSet.close()

def svm3d(table, xmin, xmax, ymin, ymax, xresolution, yresolution, zresolution, ratio, seabed, file, name, gamma):
    # read file
    df = pd.read_csv(table)

    # add customized boundary
    df = df[(df['Easting'] >= float(xmin)) & (df['Easting'] <= float(xmax)) & (df['Northing'] >= float(ymin)) & (
                df['Northing'] <= float(ymax))]

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
    coor_trans = np.hstack((litoNp[:, :2] / float(ratio), litoNp[:, 2].reshape(-1, 1)))
    soil_class = litoNp[:, 3]

    # we create an instance of Neighbours Classifier and fit the data.
    clf = svm.SVC(kernel='rbf', probability=True, gamma=float(gamma))
    clf.fit(coor_trans, soil_class)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = litoNp[:, 0].min() - 1, litoNp[:, 0].max() + 1
    y_min, y_max = litoNp[:, 1].min() - 1, litoNp[:, 1].max() + 1
    z_min, z_max = litoNp[:, 2].min() - 1, litoNp[:, 2].max() + 1

    # Determination of the lithology matrix
    zresolution = - float(zresolution)
    vertexCols = np.arange(x_min, x_max, float(xresolution))
    vertexRows = np.arange(y_min, y_max, float(yresolution))
    vertexLays = np.arange(z_max, z_min, zresolution)
    cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
    cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
    cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    workspace = arcpy.env.workspace
    arcpy.CreateFeatureclass_management(workspace, 'Pts', "POINT", '', 'DISABLED', 'ENABLED',
                                        arcpy.SpatialReference(2326, 5738))
    cur_borept = arcpy.da.InsertCursor('Pts', "SHAPE@")

    litoMatrix = np.zeros([nLays, nRows, nCols])
    ProbMatrix = np.zeros([nLays, nRows, nCols])
    for lay in range(nLays):
        for row in range(nRows):
            for col in range(nCols):
                # predict the soil class of unknown points and the info entropy
                cellXYZ = [cellCols[col], cellRows[row], cellLays[lay]]
                cellTrans = np.array([cellCols[col] / float(ratio), cellRows[row] / float(ratio), cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)
                pt = arcpy.Point(cellCols[col], cellRows[row], cellLays[lay])
                cur_borept.insertRow([pt])

    arcpy.sa.ExtractValuesToPoints("Pts", seabed, "pt_on_seabed", "None", "VALUE_ONLY")

    # store value of each points
    value_seabed = []
    pt_seabed = arcpy.da.SearchCursor("pt_on_seabed", ['SHAPE@XY', 'SHAPE@Z', 'RASTERVALU'])
    for row in pt_seabed:
        if row[2] is not None and row[1] >= row[2]:
            value_seabed.append([format(row[0][0], '.1f'), format(row[0][1], '.1f'), format(row[1], '.1f')])

    # add water layer
    for lay in range(nLays):
        for row in range(nRows):
            for col in range(nCols):
                cellXYZ = [format(cellCols[col], '.1f'), format(cellRows[row], '.1f'), format(cellLays[lay], '.1f')]
                if cellXYZ in value_seabed:
                    litoMatrix[lay, row, col] = 0
                    ProbMatrix[lay, row, col] = ma.masked
    arcpy.AddMessage(litoMatrix)

    del cur_borept
    del pt_seabed
    arcpy.management.Delete("Pts")
    arcpy.management.Delete("Pts")
    arcpy.management.Delete("pt_on_seabed")
    arcpy.management.Delete("pt_on_seabed")

    # Create NetCDF
    name = name + '.nc'
    outDataSet = netCDF4.Dataset(os.path.join(file, name), 'w', format='NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z', nLays)
    outDataSet.createDimension('y', nRows)
    outDataSet.createDimension('x', nCols)

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology = outDataSet.createVariable('Lithology', int, ('z', 'y', 'x'), fill_value=-9999)
    info_entro = outDataSet.createVariable('Information Entropy', np.float32, ('z', 'y', 'x'), fill_value=-9999)

    # Assign values
    ncX[:] = cellCols
    ncY[:] = cellRows
    ncZ[:] = cellLays

    ncLithology[:, :, :] = litoMatrix
    info_entro[:, :, :] = ProbMatrix

    ncLithology.long_name = 'Lithology'
    info_entro.long_name = 'Entropy'

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

    outDataSet.close()

def gbdt3d(table, xmin, xmax, ymin, ymax, xresolution, yresolution, zresolution, ratio, seabed, file, name, gbdt_n):
    # read file
    df = pd.read_csv(table)

    # add customized boundary
    df = df[(df['Easting'] >= float(xmin)) & (df['Easting'] <= float(xmax)) & (df['Northing'] >= float(ymin)) & (
                df['Northing'] <= float(ymax))]

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
    coor_trans = np.hstack((litoNp[:, :2] / float(ratio), litoNp[:, 2].reshape(-1, 1)))
    soil_class = litoNp[:, 3]

    # we create an instance of Neighbours Classifier and fit the data.
    clf = GradientBoostingClassifier(n_estimators=int(gbdt_n))
    clf.fit(coor_trans, soil_class)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = litoNp[:, 0].min() - 1, litoNp[:, 0].max() + 1
    y_min, y_max = litoNp[:, 1].min() - 1, litoNp[:, 1].max() + 1
    z_min, z_max = litoNp[:, 2].min() - 1, litoNp[:, 2].max() + 1

    # Determination of the lithology matrix
    zresolution = - float(zresolution)
    vertexCols = np.arange(x_min, x_max, float(xresolution))
    vertexRows = np.arange(y_min, y_max, float(yresolution))
    vertexLays = np.arange(z_max, z_min, zresolution)
    cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
    cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
    cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    workspace = arcpy.env.workspace
    arcpy.CreateFeatureclass_management(workspace, 'Pts', "POINT", '', 'DISABLED', 'ENABLED',
                                        arcpy.SpatialReference(2326, 5738))
    cur_borept = arcpy.da.InsertCursor('Pts', "SHAPE@")

    litoMatrix = np.zeros([nLays, nRows, nCols])
    ProbMatrix = np.zeros([nLays, nRows, nCols])
    for lay in range(nLays):
        for row in range(nRows):
            for col in range(nCols):
                # predict the soil class of unknown points and the info entropy
                cellXYZ = [cellCols[col], cellRows[row], cellLays[lay]]
                cellTrans = np.array([cellCols[col] / float(ratio), cellRows[row] / float(ratio), cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)
                pt = arcpy.Point(cellCols[col], cellRows[row], cellLays[lay])
                cur_borept.insertRow([pt])

    arcpy.sa.ExtractValuesToPoints("Pts", seabed, "pt_on_seabed", "None", "VALUE_ONLY")

    # store value of each points
    value_seabed = []
    pt_seabed = arcpy.da.SearchCursor("pt_on_seabed", ['SHAPE@XY', 'SHAPE@Z', 'RASTERVALU'])
    for row in pt_seabed:
        if row[2] is not None and row[1] >= row[2]:
            value_seabed.append([format(row[0][0], '.1f'), format(row[0][1], '.1f'), format(row[1], '.1f')])

    # add water layer
    for lay in range(nLays):
        for row in range(nRows):
            for col in range(nCols):
                cellXYZ = [format(cellCols[col], '.1f'), format(cellRows[row], '.1f'), format(cellLays[lay], '.1f')]
                if cellXYZ in value_seabed:
                    litoMatrix[lay, row, col] = 0
                    ProbMatrix[lay, row, col] = ma.masked
    arcpy.AddMessage(litoMatrix)

    del cur_borept
    del pt_seabed
    arcpy.management.Delete("Pts")
    arcpy.management.Delete("Pts")
    arcpy.management.Delete("pt_on_seabed")
    arcpy.management.Delete("pt_on_seabed")

    # Create NetCDF
    name = name + '.nc'
    outDataSet = netCDF4.Dataset(os.path.join(file, name), 'w', format='NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z', nLays)
    outDataSet.createDimension('y', nRows)
    outDataSet.createDimension('x', nCols)

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology = outDataSet.createVariable('Lithology', int, ('z', 'y', 'x'), fill_value=-9999)
    info_entro = outDataSet.createVariable('Information Entropy', np.float32, ('z', 'y', 'x'), fill_value=-9999)

    # Assign values
    ncX[:] = cellCols
    ncY[:] = cellRows
    ncZ[:] = cellLays

    ncLithology[:, :, :] = litoMatrix
    info_entro[:, :, :] = ProbMatrix

    ncLithology.long_name = 'Lithology'
    info_entro.long_name = 'Entropy'

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

    outDataSet.close()

def rf3d(table, xmin, xmax, ymin, ymax, xresolution, yresolution, zresolution, ratio, seabed, file, name, rf_n):
    # read file
    df = pd.read_csv(table)

    # add customized boundary
    df = df[(df['Easting'] >= float(xmin)) & (df['Easting'] <= float(xmax)) & (df['Northing'] >= float(ymin)) & (
                df['Northing'] <= float(ymax))]

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
    coor_trans = np.hstack((litoNp[:, :2] / float(ratio), litoNp[:, 2].reshape(-1, 1)))
    soil_class = litoNp[:, 3]

    # we create an instance of Neighbours Classifier and fit the data.
    clf = RandomForestClassifier(int(rf_n))
    clf.fit(coor_trans, soil_class)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = litoNp[:, 0].min() - 1, litoNp[:, 0].max() + 1
    y_min, y_max = litoNp[:, 1].min() - 1, litoNp[:, 1].max() + 1
    z_min, z_max = litoNp[:, 2].min() - 1, litoNp[:, 2].max() + 1

    # Determination of the lithology matrix
    zresolution = - float(zresolution)
    vertexCols = np.arange(x_min, x_max, float(xresolution))
    vertexRows = np.arange(y_min, y_max, float(yresolution))
    vertexLays = np.arange(z_max, z_min, zresolution)
    cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
    cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
    cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
    nCols = cellCols.shape[0]
    nRows = cellRows.shape[0]
    nLays = cellLays.shape[0]

    workspace = arcpy.env.workspace
    arcpy.CreateFeatureclass_management(workspace, 'Pts', "POINT", '', 'DISABLED', 'ENABLED',
                                        arcpy.SpatialReference(2326, 5738))
    cur_borept = arcpy.da.InsertCursor('Pts', "SHAPE@")

    litoMatrix = np.zeros([nLays, nRows, nCols])
    ProbMatrix = np.zeros([nLays, nRows, nCols])
    for lay in range(nLays):
        for row in range(nRows):
            for col in range(nCols):
                # predict the soil class of unknown points and the info entropy
                cellXYZ = [cellCols[col], cellRows[row], cellLays[lay]]
                cellTrans = np.array([cellCols[col] / float(ratio), cellRows[row] / float(ratio), cellLays[lay]])
                litoMatrix[lay, row, col] = clf.predict([cellTrans])
                prob_pre = clf.predict_proba([cellTrans])
                ProbMatrix[lay, row, col] = entropy(prob_pre.flatten(), base=2)
                pt = arcpy.Point(cellCols[col], cellRows[row], cellLays[lay])
                cur_borept.insertRow([pt])

    arcpy.sa.ExtractValuesToPoints("Pts", seabed, "pt_on_seabed", "None", "VALUE_ONLY")

    # store value of each points
    value_seabed = []
    pt_seabed = arcpy.da.SearchCursor("pt_on_seabed", ['SHAPE@XY', 'SHAPE@Z', 'RASTERVALU'])
    for row in pt_seabed:
        if row[2] is not None and row[1] >= row[2]:
            value_seabed.append([format(row[0][0], '.1f'), format(row[0][1], '.1f'), format(row[1], '.1f')])

    # add water layer
    for lay in range(nLays):
        for row in range(nRows):
            for col in range(nCols):
                cellXYZ = [format(cellCols[col], '.1f'), format(cellRows[row], '.1f'), format(cellLays[lay], '.1f')]
                if cellXYZ in value_seabed:
                    litoMatrix[lay, row, col] = 0
                    ProbMatrix[lay, row, col] = ma.masked
    arcpy.AddMessage(litoMatrix)

    del cur_borept
    del pt_seabed
    arcpy.management.Delete("Pts")
    arcpy.management.Delete("Pts")
    arcpy.management.Delete("pt_on_seabed")
    arcpy.management.Delete("pt_on_seabed")

    # Create NetCDF
    name = name + '.nc'
    outDataSet = netCDF4.Dataset(os.path.join(file, name), 'w', format='NETCDF4')

    # Create dimensions
    outDataSet.createDimension('z', nLays)
    outDataSet.createDimension('y', nRows)
    outDataSet.createDimension('x', nCols)

    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))

    ncLithology = outDataSet.createVariable('Lithology', int, ('z', 'y', 'x'), fill_value=-9999)
    info_entro = outDataSet.createVariable('Information Entropy', np.float32, ('z', 'y', 'x'), fill_value=-9999)

    # Assign values
    ncX[:] = cellCols
    ncY[:] = cellRows
    ncZ[:] = cellLays

    ncLithology[:, :, :] = litoMatrix
    info_entro[:, :, :] = ProbMatrix

    ncLithology.long_name = 'Lithology'
    info_entro.long_name = 'Entropy'

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

    outDataSet.close()


#PARAMETER
tb = arcpy.GetParameterAsText(0)
xmin = arcpy.GetParameterAsText(1)
xmax = arcpy.GetParameterAsText(2)
ymin = arcpy.GetParameterAsText(3)
ymax = arcpy.GetParameterAsText(4)
xresolution = arcpy.GetParameterAsText(5)
yresolution = arcpy.GetParameterAsText(6)
zresolution = arcpy.GetParameterAsText(7)
ratio = arcpy.GetParameterAsText(8)
seabed= arcpy.GetParameterAsText(9)
file = arcpy.GetParameterAsText(10)
name = arcpy.GetParameterAsText(11)
method = arcpy.GetParameterAsText(12)
neighbour = arcpy.GetParameterAsText(13)
gamma = arcpy.GetParameterAsText(14)
gbdt_n = arcpy.GetParameterAsText(15)
rf_n = arcpy.GetParameterAsText(16)
if method == 'kNN':
    knn3d(tb, xmin, xmax, ymin, ymax, xresolution, yresolution, zresolution, ratio, seabed, file, name, neighbour)
elif method == 'SVM':
    svm3d(tb, xmin, xmax, ymin, ymax, xresolution, yresolution, zresolution, ratio, seabed, file, name, gamma)
elif method == 'GBDT':
    gbdt3d(tb, xmin, xmax, ymin, ymax, xresolution, yresolution, zresolution, ratio, seabed, file, name, gbdt_n)
elif method == 'RF':
    gbdt3d(tb, xmin, xmax, ymin, ymax, xresolution, yresolution, zresolution, ratio, seabed, file, name, rf_n)
else:
    pass
