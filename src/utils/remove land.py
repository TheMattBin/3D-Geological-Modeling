import numpy as np
import numpy.ma as ma

import rasterio
import netCDF4 as nc

def load_nc_dataset(nc_path: str) -> nc.Dataset:
    """
    Load a NetCDF dataset from the given path.
    """
    return nc.Dataset(nc_path)

def load_raster(raster_path: str) -> rasterio.io.DatasetReader:
    """
    Load a raster file from the given path.
    """
    return rasterio.open(raster_path)

def model_land_no_(var: nc.Dataset, seabed_raster: rasterio.io.DatasetReader, shenzhen_raster: rasterio.io.DatasetReader, output_nc: str = "removeland.nc") -> None:
    """
    Mask land and Shenzhen area in lithology NetCDF and save to new NetCDF file.
    """
    litoMatrix = var['Lithology'][:]
    X = var['x'][:]
    Y = var['y'][:]
    Z = var['z'][:]
    north = litoMatrix.shape[1]
    east = litoMatrix.shape[2]
    land = []
    for j in range(north):
        for i in range(east):
            if 800576.1164593603 <= X[i] <= 870476.1164593603 and 834121.0385106392 <= Y[j] <= 858521.0385106392:
                if shenzhen_raster.index(X[i], Y[j]):
                    land.append([j, i])
            if 801975.0 <= X[i] <= 860025.0 and 800975.0 <= Y[j] <= 847525.0:
                x, y = seabed_raster.index(X[i], Y[j])
                val = seabed_raster.read(1)[x, y]
                if val <= -3.4e+38:
                    land.append([j, i])
    for l in land:
        litoMatrix[:, l[0], l[1]] = 12
    _write_lithology_nc(output_nc, litoMatrix, X, Y, Z, var)

def model_land_no_SZ(var: nc.Dataset, seabed_raster: rasterio.io.DatasetReader, output_nc: str = "removeland_noZHv2.nc") -> None:
    """
    Mask land and out-of-HK area in lithology NetCDF and save to new NetCDF file.
    """
    litoMatrix = var['Lithology'][:]
    X = var['x'][:]
    Y = var['y'][:]
    Z = var['z'][:]
    north = litoMatrix.shape[1]
    east = litoMatrix.shape[2]
    land = []
    out_hk = []
    for j in range(north):
        for i in range(east):
            if 801975.0 <= X[i] <= 860025.0 and 800975.0 <= Y[j] <= 847525.0:
                x, y = seabed_raster.index(X[i], Y[j])
                val = seabed_raster.read(1)[x, y]
                if val <= -3.4e+38:
                    land.append([j, i])
            else:
                out_hk.append([j, i])
    for l in land:
        litoMatrix[:, l[0], l[1]] = 12
    for l in out_hk:
        litoMatrix[:, l[0], l[1]] = 13
    _write_lithology_nc(output_nc, litoMatrix, X, Y, Z, var, add_esri_pe_string=True)

def model_hide_land(var: nc.Dataset, output_nc: str = "removeland_hide_all_land.nc") -> None:
    """
    Mask all land and out-of-HK areas in lithology NetCDF and save to new NetCDF file.
    """
    litoMatrix = var['Lithology'][:]
    X = var['x'][:]
    Y = var['y'][:]
    Z = var['z'][:]
    depth = litoMatrix.shape[0]
    north = litoMatrix.shape[1]
    east = litoMatrix.shape[2]
    for k in range(depth):
        for j in range(north):
            for i in range(east):
                if litoMatrix[k, j, i] == 12 or litoMatrix[k, j, i] == 13:
                    litoMatrix[k, j, i] = ma.masked
    _write_lithology_nc(output_nc, litoMatrix, X, Y, Z, var, dtype='i1', add_esri_pe_string=True)

def _write_lithology_nc(
    output_nc: str,
    litoMatrix: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    ref_var: nc.Dataset,
    dtype: str = 'i4',
    add_esri_pe_string: bool = False
) -> None:
    """
    Helper to write a lithology NetCDF file.
    """
    outDataSet = nc.Dataset(output_nc, 'w', format='NETCDF4')
    outDataSet.createDimension('z', len(Z))
    outDataSet.createDimension('y', len(Y))
    outDataSet.createDimension('x', len(X))
    ncZ = outDataSet.createVariable('z', np.float32, ('z',))
    ncY = outDataSet.createVariable('y', np.float32, ('y',))
    ncX = outDataSet.createVariable('x', np.float32, ('x',))
    ncLithology = outDataSet.createVariable('Lithology', dtype, ('z', 'y', 'x'), fill_value=-9999)
    ncX[:] = X
    ncY[:] = Y
    ncZ[:] = Z
    ncLithology[:, :, :] = litoMatrix
    ncLithology.long_name = 'Lithology'
    ncZ.positive = 'down'
    ncY.standard_name = 'projection_y_coordinate'
    ncY.units = 'm'
    ncX.standard_name = 'projection_x_coordinate'
    ncX.units = 'm'
    if add_esri_pe_string:
        outDataSet.esri_pe_string = (
            'PROJCS["Hong_Kong_1980_Grid",GEOGCS["GCS_Hong_Kong_1980",'
            'DATUM["D_Hong_Kong_1980",SPHEROID["International_1924",6378388.0,297.0]],'
            'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
            'PARAMETER["False_Easting",836694.05],PARAMETER["False_Northing",819069.8],PARAMETER["Central_Meridian",114.1785555555556],'
            'PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",22.31213333333334],UNIT["Meter",1.0]],'
            'VERTCS["Hong_Kong_Principal_Datum",VDATUM["Hong_Kong_Principal_Datum"],PARAMETER["Vertical_Shift",0.0],'
            'PARAMETER["Direction",1.0],UNIT["Meter",1.0]]'
        )
    ref_var.close()
    outDataSet.close()

def main():
    # Example usage; update the paths as needed
    nc_path = "./data/removeland_noZHv2.nc"
    seabed_raster_path = "./data/seabed_no_hk.tif"
    shenzhen_raster_path = "./data/ShenzhenArea_PolygonToRaster1.tif"
    var = load_nc_dataset(nc_path)
    seabed_raster = load_raster(seabed_raster_path)
    shenzhen_raster = load_raster(shenzhen_raster_path)
    # Uncomment the function you want to run:
    # model_land_no_(var, seabed_raster, shenzhen_raster, output_nc="removeland.nc")
    # model_land_no_SZ(var, seabed_raster, output_nc="removeland_noZHv2.nc")
    model_hide_land(var, output_nc="removeland_hide_all_land.nc")

if __name__ == "__main__":
    main()