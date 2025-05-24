from datetime import datetime
import pandas as pd
import numpy as np
import netCDF4
from pykrige.ok3d import OrdinaryKriging3D


def create_netcdf(filename, vertexCols, vertexRows, vertexLays, data):
    nCols = vertexCols.shape[0]
    nRows = vertexRows.shape[0]
    nLays = vertexLays.shape[0]
    outDataSet = netCDF4.Dataset(filename, 'w', format='NETCDF4')
    outDataSet.createDimension('z', nLays)
    outDataSet.createDimension('y', nRows)
    outDataSet.createDimension('x', nCols)
    ncZ = outDataSet.createVariable('z', np.float32, ('z'))
    ncY = outDataSet.createVariable('y', np.float32, ('y'))
    ncX = outDataSet.createVariable('x', np.float32, ('x'))
    info_entro_m = outDataSet.createVariable('SPT-N', np.float32, ('z', 'y', 'x'), fill_value=-9999)
    ncX[:] = vertexCols
    ncY[:] = vertexRows
    ncZ[:] = vertexLays
    info_entro_m[:, :, :] = data
    info_entro_m.long_name = 'SPT-N'
    ncZ.positive = 'up'
    ncY.standard_name = 'projection_y_coordinate'
    ncY.units = 'm'
    ncX.standard_name = 'projection_x_coordinate'
    ncX.units = 'm'
    outDataSet.close()


def main():
    # Parameters
    csv_path = 'SPT_CW.csv'
    variogram_model = "exponential"
    grid_spacing_xy = 1000
    grid_spacing_z = 1

    df = pd.read_csv(csv_path)
    coor = np.stack((df['Easting'].values, df['Northing'].values, df['MidElev'].values), axis=-1)
    SPTN = df['SPTN'].values

    x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
    y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
    z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1

    vertexCols = np.arange(x_min, x_max, grid_spacing_xy)
    vertexRows = np.arange(y_min, y_max, grid_spacing_xy)
    vertexLays = np.arange(z_max, z_min, -grid_spacing_z)

    ok3d = OrdinaryKriging3D(
        coor[:, 0], coor[:, 1], coor[:, 2], SPTN, variogram_model=variogram_model
    )
    k3d1, _ = ok3d.execute("grid", vertexCols, vertexRows, vertexLays)

    t = datetime.now()
    filename = f"myOutputGBC-{t.strftime('%d-%m-%Y-%H-%M-%S')}.nc"
    create_netcdf(filename, vertexCols, vertexRows, vertexLays, k3d1)


if __name__ == "__main__":
    main()