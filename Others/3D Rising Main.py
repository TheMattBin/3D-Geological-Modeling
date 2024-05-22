# -*- coding: utf-8 -*-
"""
Created on Thur Oct 28 11:22:48 2021

@author: Matthew.Liu
"""

import os
import pandas as pd
import numpy as np
import arcpy


def RisingMain(feature):
    Risingmain_name = os.path.basename(feature)
    cur = arcpy.da.SearchCursor(feature, ['SHAPE@', '*'])

    pipe_list = []
    for row in cur:

        # Set start point
        startpt = row[0].firstPoint

        # Set Start coordinates
        startx = startpt.X
        starty = startpt.Y

        # Set end point
        endpt = row[0].lastPoint

        # Set End coordinates
        endx = endpt.X
        endy = endpt.Y

        tmp = (startx, starty, endx, endy,) + row[3:]
        pipe_list.append(tmp)


    sr = arcpy.SpatialReference(2326, 5738)
    arcpy.management.CreateFeatureclass(r'D:\HK Model Arcgis\HK Model ArcPro\Drianage Pip 3D', Risingmain_name, 'POLYLINE', '', 'ENABLED', 'ENABLED', sr)
    arcpy.management.AddField(Risingmain_name, 'OBJECTID', 'SHORT')
    arcpy.management.AddField(Risingmain_name, 'WIDTH', 'SHORT')
    arcpy.management.AddField(Risingmain_name, 'FEAT_NUM', 'TEXT')
    arcpy.management.AddField(Risingmain_name, 'SHAPE_1', 'TEXT')
    arcpy.management.AddField(Risingmain_name, 'US_IL', 'DOUBLE')
    arcpy.management.AddField(Risingmain_name, 'DS_IL', 'DOUBLE')

    RMSW_cur = arcpy.da.InsertCursor(Risingmain_name, "SHAPE@")
    for i in range(len(pipe_list)):
        RMSW_array = arcpy.Array( [arcpy.Point(pipe_list[i][0], pipe_list[i][1], pipe_list[i][-2]), arcpy.Point(pipe_list[i][2], pipe_list[i][3], pipe_list[i][-1])])
        polyline_RMSW = arcpy.Polyline(RMSW_array, None, True)
        RMSW_cur.insertRow([polyline_RMSW])
        RMSW_array.removeAll()

    del RMSW_cur
    del cur

    i = 0
    cur_update = arcpy.da.UpdateCursor(Risingmain_name,['OBJECTID', 'WIDTH', 'FEAT_NUM', 'SHAPE_1', 'US_IL', 'DS_IL'])


    for row in cur_update:
        row = pipe_list[i][4:]
        cur_update.updateRow(row)
        i += 1

    del cur_update

#Parameters
fc = arcpy.GetParameterAsText(0)
fc = fc.split(';')
print(fc)
for f in fc:
    RisingMain(fc)