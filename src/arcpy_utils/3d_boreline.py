import arcpy

#can add 3D length of borehole
#use points to create polylines
def boreline(spatial_ref, bore_table, workplace):
    try:
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
            polyline_bore = arcpy.Polyline(Bore_array, None, True)
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

        for row in cur_update:
            row = soil_int[i]
            cur_update.updateRow(row)
            i += 1

        del cur_update

    except:
        pass

#PARAMETER
fc = arcpy.GetParameterAsText(0)
bore_table = arcpy.GetParameterAsText(1)
workplace = arcpy.GetParameterAsText(2)
try:
    boreline(fc, bore_table, workplace)
except:
    pass
