import arcpy

#can add 3D length of borehole
#use points to create polylines
def boreline(spatial_ref, bore_table, workplace):
    try:
        #fc = r"D:\HK Model Arcgis\HK Model ArcPro\trail2\trail2.gdb\MHags_V61_XYTableToPoint12"
        boreTB = 'BoreholeTable'
        arcpy.conversion.TableToTable(bore_table, workplace, boreTB)
        arcpy.management.CreateFeatureclass(workplace, 'Boreline',
                                            'POLYLINE', '', 'ENABLED', 'ENABLED', spatial_ref)
        arcpy.management.AddField('Boreline', 'BHID', 'TEXT')
        arcpy.management.AddField('Boreline', 'LocationType', 'TEXT')
        arcpy.management.AddField('Boreline', 'Easting', 'DOUBLE')
        arcpy.management.AddField('Boreline', 'Northing', 'DOUBLE')
        arcpy.management.AddField('Boreline', 'Depth', 'DOUBLE')
        arcpy.management.AddField('Boreline', 'GroundLevel', 'DOUBLE')
        arcpy.management.AddField('Boreline', 'TopElev', 'DOUBLE')
        arcpy.management.AddField('Boreline', 'BotElev', 'DOUBLE')
        arcpy.management.AddField('Boreline', 'Soil_type', 'TEXT')
        arcpy.management.AddField('Boreline', 'Lithology', 'TEXT')
        arcpy.management.AddField('Boreline', 'GeoCode', 'SHORT')
        boreline_cur = arcpy.da.InsertCursor('Boreline', "SHAPE@")

        cur_borept = arcpy.da.SearchCursor(boreTB, '*')
        borept_list = []
        for row in cur_borept:
            borept_list.append(row)

        for i in range(len(borept_list)-1):
            Bore_array = arcpy.Array( [arcpy.Point(borept_list[i][3], borept_list[i][4], borept_list[i][7]), arcpy.Point(borept_list[i][3], borept_list[i][4], borept_list[i][8])])
            # print([bp[2], blog[4], blog[5]])
            # print(list(Bore_array))
            polyline_bore = arcpy.Polyline(Bore_array, None, True)
            # print(list(polyline_bore))
            boreline_cur.insertRow([polyline_bore])
            Bore_array.removeAll()

        del boreline_cur
        del cur_borept

        i = 0
        cur_update = arcpy.da.UpdateCursor('Boreline',
                                           ['BHID', 'LocationType', 'Easting', 'Northing', 'Depth', 'GroundLevel', 'TopElev', 'BotElev', 'Soil_type', 'Lithology', 'GeoCode'])
        soil_int = []
        for bp in borept_list:
            soil_int.append(bp[1:])
        #print(soil_int[0])

        for row in cur_update:
            row = soil_int[i]
            cur_update.updateRow(row)
            i += 1

        del cur_update

        '''
        fields = [loc, x, y, z, geocode]
        wellID = []
        X = []
        Y = []
        Z = []
        l = []
        l3d_fc = []
        bhid = []
        geocode2 = []
        geocode2_new = []
        

    # the feature's x,y, z coordinates
        with arcpy.da.SearchCursor(fc, fields) as cursor:
            for row in cursor:
                wellID.append(row[0])
                X.append(row[1])
                Y.append(row[2])
                Z.append(row[3])
                geocode2.append(row[4])
                #print('{0}, {1}, {2}'.format(row[0], row[1], row[2]))
        for i in range(len(X)-1):
            if wellID[i] == wellID[i+1]:
                bhid.append(wellID[i])
                geocode2_new.append(geocode2[i])
                array = arcpy.Array([arcpy.Point(X[i], Y[i], Z[i]),arcpy.Point(X[i+1], Y[i+1], Z[i+1])])
                polyline = arcpy.Polyline(array, spatial_ref, True)
                l3d = polyline.length3D
                l.append(polyline)
                l3d_fc.append(l3d)
        p1 = arcpy.management.CreateFeatureclass(workplace, 'Boreline',
                                             'POLYLINE', '', 'ENABLED', 'ENABLED', spatial_ref)
        arcpy.CopyFeatures_management(l, p1)
        #p1 = r'D:\HK Model Arcgis\HK Model ArcPro\trail2\trail2.gdb\polyline1'
        arcpy.AddField_management(p1, 'WellID', "TEXT", None, None)
        arcpy.AddField_management(p1, 'Length3d', "DOUBLE", None, None)
        arcpy.AddField_management(p1, 'GeoCode', "TEXT", None, None)

        i = 0
        with arcpy.da.UpdateCursor(p1, ['WellID', 'Length3d', 'GeoCode']) as cur:
            for row in cur:
                row[0] = bhid[i]
                row[1] = l3d_fc[i]
                row[2] = geocode2_new[i]
                cur.updateRow(row)
                i = i+1
        '''

    except:
        pass

#PARAMETER
fc = arcpy.GetParameterAsText(0)
bore_table = arcpy.GetParameterAsText(1)
workplace = arcpy.GetParameterAsText(2)
try:
    boreline(fc, bore_table, workplace)
    #boreline(fc, loc, X, Y, Z, geocode, workplace)
except:
    pass
