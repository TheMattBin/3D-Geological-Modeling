import arcpy
import numpy as np
import netCDF4
from collections import defaultdict

def CrossSection_3D_Model(geo_model, polyline_input, name, y_cellsize):
	# load model can get (x,y,z) as array
	voxel = netCDF4.Dataset(geo_model)
	east = voxel['x'][:]
	northing = voxel['y'][:]
	depth = voxel['z'][:]

	# Create a fishnet
	workspace = arcpy.env.workspace
	originCoordinate = str(min(east)) + ' ' + str(min(northing))
	yAxisCoordinate = str(min(east)) + ' ' + str(min(northing) + 0.1)
	cellSizeWidth = str(east[1] - east[0])
	cellSizeHeight = str(northing[1] - northing[0])
	cellSizeDepth = depth[1] - depth[0]
	oppositeCoorner = str(max(east)) + ' ' + str(max(northing))
	outFeatureClass = "fishnet_voxel.shp"
	arcpy.CreateFishnet_management(outFeatureClass, originCoordinate, yAxisCoordinate, cellSizeWidth, cellSizeHeight, '#', '#', oppositeCoorner, 'LABELS', '#', 'POLYGON')

	# Select the polygon that polyline lies in and intersect
	# import polyline to be interpret
	fishnet_selected = arcpy.arcpy.SelectLayerByLocation_management(outFeatureClass, 'INTERSECT', polyline_input)
	outFeatureClass_label = "fishnet_voxel_label.shp"
	inFeatures = [fishnet_selected, outFeatureClass_label]
	intersectOutput = "Point_for_section"
	arcpy.Intersect_analysis(inFeatures, intersectOutput)

	# Search for points of cross section
	xycursor = arcpy.da.SearchCursor(intersectOutput, 'SHAPE@XY')
	point_coord = []
	for row in xycursor:
		point_coord.append(row[0])

	# Obtain (x,y) of section line
	# Sort the points according to the first point of line
	line_cur = arcpy.da.SearchCursor(polyline_input, "SHAPE@")
	for row in line_cur:
		startpt = row[0].firstPoint
		# Set Start coordinates
		startx = startpt.X
		starty = startpt.Y
	sortarr = sorted(point_coord, key=lambda arr: (arr[0] - startx) ** 2 + (arr[1] - starty) ** 2)

	depth_knn = []
	coor_df_east = []
	coor_df_north = []
	predict_list = []
	predict_list_entropy = []
	code = defaultdict(list)
	entropy = defaultdict(list)

	def BHvsPre(var, east, north):
		# getting dimensions and soil layers
		litoMatrix = var['Lithology'][:]
		EntropyMatrix = var['Information Entropy'][:]
		X = var['x'][:]
		Y = var['y'][:]
		Z = var['z'][:]
		idx_east = (np.abs(X - east)).argmin()
		idx_north = (np.abs(Y - north)).argmin()

		predict_bh = litoMatrix[:, idx_north, idx_east]
		predict_entropy = EntropyMatrix[:, idx_north, idx_east]
		if predict_bh.size != 0:
			depth_knn.append(var['z'][:])
			predict_list.append(predict_bh)
			predict_list_entropy.append(predict_entropy)
			coor_df_east.append([east] * (len(predict_bh)))
			coor_df_north.append([north] * (len(predict_bh)))
			code[(east, north)].append(predict_bh)
			entropy[(east, north)].append(predict_entropy)

	# Obtain array from voxel layer
	for e, n in sortarr:
		BHvsPre(voxel, e, n)

	# Convert geological profile to polygon
	section_plane = []
	for key, value in code.items():
		section_plane.append(code[key][:])
	section_plane_trans = np.transpose(np.array(section_plane)).reshape((-1, len(coor_df_north)))

	# get entropy raster
	entropy_plane = []
	for key, value in entropy.items():
		entropy_plane.append(entropy[key][:])
	entropy_plane_trans = np.transpose(np.array(entropy_plane)).reshape((-1, len(coor_df_north)))
	entropy_plane_trans = np.maximum(entropy_plane_trans, 0)

	CrossSection_raster = arcpy.NumPyArrayToRaster(section_plane_trans, x_cell_size=float(cellSizeWidth), y_cell_size=float(y_cellsize))
	arcpy.RasterToPolygon_conversion(CrossSection_raster, name)

	name_entropy = name + '_entropy'
	CrossSection_raster_entropy = arcpy.NumPyArrayToRaster(entropy_plane_trans, x_cell_size=float(cellSizeWidth), y_cell_size=float(y_cellsize))
	CrossSection_raster_entropy.save(name_entropy)

	arcpy.management.Delete("fishnet_voxel.shp")
	arcpy.management.Delete("fishnet_voxel.shp")
	arcpy.management.Delete("fishnet_voxel_label.shp")
	arcpy.management.Delete("fishnet_voxel_label.shp")
	arcpy.management.Delete("Point_for_section")
	arcpy.management.Delete("Point_for_section")
	arcpy.management.Delete(CrossSection_raster)

	del xycursor
	del line_cur


#PARAMETERS
geomodel = arcpy.GetParameterAsText(0)
sectionline = arcpy.GetParameterAsText(1)
name = arcpy.GetParameterAsText(2)
ycellsize = arcpy.GetParameterAsText(3)

#RUN FUNCTION
CrossSection_3D_Model(geomodel, sectionline, name, ycellsize)
