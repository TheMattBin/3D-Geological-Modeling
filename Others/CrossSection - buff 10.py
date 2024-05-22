import arcpy
import os


# -----------------------------------------------------------------------------------------------------#

def XSection(sectionline, raster, workplace, borepoint, buff_dist, he, ve):
    X_cur_id = arcpy.da.SearchCursor(sectionline, "OID@")
    section_id = []
    for row in X_cur_id:
        section_id.append(row[0])
    del X_cur_id

    # table name maybe need to use os.path
    sectionline_table = os.path.basename(sectionline) + '_Table'
    rasterprofile = arcpy.ddd.StackProfile(sectionline, raster, sectionline_table)
    Xcur = arcpy.da.SearchCursor(sectionline_table, ['FIRST_DIST', 'FIRST_Z', 'SRC_NAME', 'SRC_NAME'])
    Xcur_list = [i for i in Xcur]

    # create points and polyline in a list
    array_section = arcpy.Array()
    for i in range(len(Xcur_list)):
        array_section.add(arcpy.Point(Xcur_list[i][0] * float(he), Xcur_list[i][1] * float(ve)))
    # print(list(array_section))
    polyline_section = arcpy.Polyline(array_section)
    # print(list(polyline_section))

    # create feature and feature name
    # use os.path
    sectionline_name = os.path.basename(sectionline) + str(section_id.pop()) + '_Profile'
    arcpy.CreateFeatureclass_management(workplace, sectionline_name, "POLYLINE")
    arcpy.management.AddField(sectionline_name, 'BHID', 'TEXT')
    arcpy.management.AddField(sectionline_name, 'Easting', 'DOUBLE')
    arcpy.management.AddField(sectionline_name, 'Northing', 'DOUBLE')
    arcpy.management.AddField(sectionline_name, 'Depth', 'DOUBLE')
    arcpy.management.AddField(sectionline_name, 'GroundLevel', 'DOUBLE')
    arcpy.management.AddField(sectionline_name, 'TopElev', 'DOUBLE')
    arcpy.management.AddField(sectionline_name, 'BotElev', 'DOUBLE')
    arcpy.management.AddField(sectionline_name, 'Soil_type', 'TEXT')
    arcpy.management.AddField(sectionline_name, 'Lithology', 'TEXT')

    # select pts within buffer
    arcpy.analysis.Buffer(sectionline, 'sectionbuff', buff_dist, "FULL", "FLAT", "NONE", '', "PLANAR")
    arcpy.management.SelectLayerByLocation(borepoint, "COMPLETELY_WITHIN", "sectionbuff")
    bore_list = []
    bore_int_list = []
    bore_cur = arcpy.da.SearchCursor(borepoint, "*")
    for row in bore_cur:
        bore_list.append(row)
    # [(0, (821926.48, 815848.66), u'R_19722 H_BH-02', 821926.48, 815848.66, 1000, 0.0, -1.9),
    # (1, (821926.48, 815848.66), u'R_19722 H_BH-02', 821926.48, 815848.66, 1000, 4.95, -6.85)]

    # append (x,y) of selected pts into a list
    Xpt_cur = arcpy.da.SearchCursor(sectionline, "SHAPE@")
    for row in Xpt_cur:
        startpt = row[0].firstPoint
        # Set Start coordinates
        startx = startpt.X
        starty = startpt.Y

        # Set end point
        # endpt = row[0].lastPoint
        # Set End coordinates
        # endx = endpt.X
        # endy = endpt.Y

    # create list of horizontal axis for selected point
    measure_list = []
    for bp in bore_list:
        tmp_dist = ((bp[1][0] - startx) ** 2 + (bp[1][1] - starty) ** 2) ** 0.5
        measure_list.append(tmp_dist)

    # Insert boreline and follow by cross-section
    Feature_cur = arcpy.da.InsertCursor(sectionline_name, "SHAPE@")
    for i in range(len(bore_list)):
        Bore_array = arcpy.Array(
            [arcpy.Point(measure_list[i] * float(he), bore_list[i][7] * float(ve)), arcpy.Point(measure_list[i] * float(he), bore_list[i][8] * float(ve))])
        polyline_bore = arcpy.Polyline(Bore_array)
        #print(list(polyline_bore))
        Feature_cur.insertRow([polyline_bore])
        Bore_array.removeAll()

    del Xcur
    del bore_cur
    del Xpt_cur
    del Feature_cur

    i = 0
    cur_update = arcpy.da.UpdateCursor(sectionline_name,
                                       ['BHID', 'Easting', 'Northing', 'Depth', 'GroundLevel', 'TopElev', 'BotElev', 'Soil_type', 'Lithology'])
    #raster_list = [(Xcur_list[0][-1], None, None, None, None, None, None, Xcur_list[0][-1], Xcur_list[0][-1])]
    soil_int = [i[2:11] for i in bore_list]
    #soil_int += raster_list
    # print(soil_int)
    for row in cur_update:
        row = soil_int[i]
        cur_update.updateRow(row)
        i += 1

    del cur_update

    sectionline_name_buff = sectionline_name + '_poly'
    arcpy.analysis.Buffer(sectionline_name, sectionline_name_buff, '10', 'FULL', 'FLAT', 'NONE', '', 'PLANAR')

    Feature_cur_section = arcpy.da.InsertCursor(sectionline_name, "SHAPE@")
    Feature_cur_section.insertRow([polyline_section])
    array_section.removeAll()

    del Feature_cur_section


    '''
    i = 0
    cur_update = arcpy.da.UpdateCursor(sectionline_name,
                                       ['BHID', 'Easting', 'Northing', 'Depth', 'GroundLevel', 'TopElev', 'BotElev', 'Soil_type', 'Lithology'])
    raster_list = [(Xcur_list[0][-1], None, None, None, None, None, None, Xcur_list[0][-1], Xcur_list[0][-1])]
    soil_int = [i[2:11] for i in bore_list]
    soil_int += raster_list
    # print(soil_int)
    for row in cur_update:
        row = soil_int[i]
        cur_update.updateRow(row)
        i += 1
    '''



    # delete run two times
    # dir provided so can del permenantly
    arcpy.management.Delete("sectionbuff")
    arcpy.management.Delete("sectionbuff")
    arcpy.management.Delete(sectionline_table)
    arcpy.management.Delete(sectionline_table)

#-----------------------------------------------------------------------------------------------------#

#PARAMETERS
sectionline = arcpy.GetParameterAsText(0)
raster = arcpy.GetParameterAsText(1)
workplace = arcpy.GetParameterAsText(2)
borepoint = arcpy.GetParameterAsText(3)
buff_dist = arcpy.GetParameterAsText(4)
he = arcpy.GetParameterAsText(5)
ve = arcpy.GetParameterAsText(6)

#-----------------------------------------------------------------------------------------------------#

#RUN FUNCTION
XSection(sectionline, raster, workplace, borepoint, buff_dist, he, ve)
