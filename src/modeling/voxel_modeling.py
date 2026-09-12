"""
====================================================================
3D Geological/Geotechnical Model using Machine Learning
It will plot the decision boundaries for each soil/lithology class.
Export as NetCDF file for further analysis.
====================================================================

"""


import pandas as pd
import numpy as np
import numpy.ma as ma
import rasterio
import netCDF4

from scipy.stats import entropy
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier


class BaseVoxelModel:
    def __init__(self, dist=100, seabed_raster_path=None):
        self.dist = dist
        self.seabed_raster = rasterio.open(seabed_raster_path) if seabed_raster_path else None

    def point_cloud_trans(self, df):
        litoPoints = []
        for _, values in df.iterrows():
            wellX, wellY = values.Easting, values.Northing
            wellXY = [wellX, wellY]
            litoPoints.append(wellXY + [values['Ground Level'] - values['Depth Top'], values['Legend Code']])
            litoPoints.append(wellXY + [values['Ground Level'] - values['Depth Base'], values['Legend Code']])
            litoLength = (values['Ground Level'] - values['Depth Top']) - (values['Ground Level'] - values['Depth Base'])
            if litoLength < 1:
                midPoint = wellXY + [
                    (values['Ground Level'] - values['Depth Top']) - litoLength / 2,
                    values['Legend Code']]
                litoPoints.append(midPoint)
                continue
            npoints = int(litoLength)
            for point in range(1, npoints + 1):
                disPoint = wellXY + [
                    (values['Ground Level'] - values['Depth Top']) - litoLength * point / (npoints + 1),
                    values['Legend Code']]
                litoPoints.append(disPoint)
        litoNp = np.array(litoPoints)
        coor_trans = np.hstack((litoNp[:, :2] / self.dist, litoNp[:, 2].reshape(-1, 1)))
        soil_class = litoNp[:, 3]
        return litoNp, coor_trans, soil_class

    def _grid_queries(self, cellCols, cellRows, cellLays):
        """Transformed query coordinates for every voxel, in the original loop
        order lay -> row -> col, so a flat result reshapes directly to
        (nLays, nRows, nCols)."""
        nLays = cellLays.shape[0]
        nRows = cellRows.shape[0]
        nCols = cellCols.shape[0]
        Xq = np.empty((nLays, nRows, nCols, 3))
        Xq[..., 0] = (cellCols / self.dist)[None, None, :]
        Xq[..., 1] = (cellRows / self.dist)[None, :, None]
        Xq[..., 2] = cellLays[:, None, None]
        return Xq.reshape(-1, 3)

    def _predict_grid(self, clf, cellCols, cellRows, cellLays):
        """Batched replacement for the former per-voxel
        ``clf.predict([cellTrans])`` / ``clf.predict_proba([cellTrans])`` /
        ``entropy(prob_pre.flatten(), base=2)`` loops. One call each for
        predict, predict_proba and the row-wise entropy."""
        shape = (cellLays.shape[0], cellRows.shape[0], cellCols.shape[0])
        Xq = self._grid_queries(cellCols, cellRows, cellLays)
        lito = clf.predict(Xq).reshape(shape)
        ent = entropy(clf.predict_proba(Xq), base=2, axis=1).reshape(shape)
        return lito, ent

    def _seabed_masks(self, cellCols, cellRows):
        """Seabed elevation and nodata-land mask per (row, col). The raster
        band is read once instead of per voxel; ``index()`` is queried per
        cell exactly as before, with the same ``band[row, col]`` indexing and
        the same scalar ``<= -3.4e+38`` nodata test (kept as a per-cell
        scalar comparison — raster.nodata reads back as the float32-rounded
        -3.3999999521443642e+38, which classifies float32-nodata pixels
        differently, and numpy < 2 evaluates scalar and array comparisons
        against the -3.4e+38 literal differently)."""
        seabed = self.seabed_raster.read(1)
        nRows = cellRows.shape[0]
        nCols = cellCols.shape[0]
        val = np.empty((nRows, nCols), dtype=seabed.dtype)
        land = np.zeros((nRows, nCols), dtype=bool)
        for row in range(nRows):
            for col in range(nCols):
                r, c = self.seabed_raster.index(cellCols[col], cellRows[row])
                v = seabed[r, c]
                val[row, col] = v
                land[row, col] = v <= -3.4e+38
        return val, land

    def create_voxel(self, cellCols, cellRows, cellLays, litoMatrix, ProbMatrix, filename):
        nCols = cellCols.shape[0]
        nRows = cellRows.shape[0]
        nLays = cellLays.shape[0]
        outDataSet = netCDF4.Dataset(filename, 'w', format='NETCDF4')
        outDataSet.createDimension('z', nLays)
        outDataSet.createDimension('y', nRows)
        outDataSet.createDimension('x', nCols)
        ncZ = outDataSet.createVariable('z', np.float32, ('z'))
        ncY = outDataSet.createVariable('y', np.float32, ('y'))
        ncX = outDataSet.createVariable('x', np.float32, ('x'))
        ncLithology_m = outDataSet.createVariable('Lithology', int, ('z', 'y', 'x'), fill_value=-9999)
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

    def fit_predict(self, coor, coor_trans, soil_class):
        raise NotImplementedError

    def run(self, df, filename, **kwargs):
        litoNp, coor_trans, soil_class = self.point_cloud_trans(df)
        a, b, c, d, e = self.fit_predict(litoNp, coor_trans, soil_class, **kwargs)
        self.create_voxel(a, b, c, d, e, filename)


class KNNVoxelModel(BaseVoxelModel):
    def __init__(self, dist=100, n_neighbors=15, seabed_raster_path=None):
        super().__init__(dist, seabed_raster_path)
        self.n_neighbors = n_neighbors

    def fit_predict(self, coor, coor_trans, soil_class, **kwargs):
        clf = neighbors.KNeighborsClassifier(self.n_neighbors, weights='distance')
        clf.fit(coor_trans, soil_class)
        x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
        y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
        z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1
        vertexCols = np.arange(x_min, x_max, 100)
        vertexRows = np.arange(y_min, y_max, 100)
        vertexLays = np.arange(z_max, z_min, -1)
        cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
        cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
        cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
        nCols, nRows, nLays = cellCols.shape[0], cellRows.shape[0], cellLays.shape[0]
        litoMatrix = ma.zeros([nLays, nRows, nCols])
        ProbMatrix = ma.zeros([nLays, nRows, nCols])
        lito, ent = self._predict_grid(clf, cellCols, cellRows, cellLays)
        litoMatrix[:, :, :] = lito
        ProbMatrix[:, :, :] = ent
        if self.seabed_raster:
            val, land = self._seabed_masks(cellCols, cellRows)
            # Old per-voxel rule: mask iff cellLays[lay] >= val and
            # [cellCols[col], cellRows[row]] not in land, where land collected
            # exactly the coordinates whose pixel is nodata (appended before
            # the membership test in the same iteration, so a nodata cell is
            # always blocked). Equivalent to: valid pixel and at/above seabed.
            water = (~land)[None, :, :] & (cellLays[:, None, None] >= val[None, :, :])
            litoMatrix[water] = 0
            ProbMatrix[water] = ma.masked
        return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix


class RFVoxelModel(BaseVoxelModel):
    def __init__(self, dist=100, n_estimators=45, seabed_raster_path=None):
        super().__init__(dist, seabed_raster_path)
        self.n_estimators = n_estimators

    def fit_predict(self, coor, coor_trans, soil_class, **kwargs):
        clf = RandomForestClassifier(self.n_estimators)
        clf.fit(coor_trans, soil_class)
        x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
        y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
        z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1
        vertexCols = np.arange(x_min, x_max, 10)
        vertexRows = np.arange(y_min, y_max, 10)
        vertexLays = np.arange(z_max, z_min, -0.2)
        cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
        cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
        cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
        nCols = cellCols.shape[0]
        nRows = cellRows.shape[0]
        nLays = cellLays.shape[0]
        litoMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
        ProbMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
        lito, ent = self._predict_grid(clf, cellCols, cellRows, cellLays)
        litoMatrix[:, :, :] = lito
        ProbMatrix[:, :, :] = ent
        if self.seabed_raster:
            val, land = self._seabed_masks(cellCols, cellRows)
            # Same equivalence as KNNVoxelModel: land == nodata pixels, which
            # always fail the old "not in land" test; water == valid pixels at
            # or above the seabed elevation.
            water = (~land)[None, :, :] & (cellLays[:, None, None] >= val[None, :, :])
            litoMatrix[water] = 0
            ProbMatrix[water] = ma.masked
        return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix


class SVMVoxelModel(BaseVoxelModel):
    def __init__(self, dist=100, gamma=0.5, seabed_raster_path=None):
        super().__init__(dist, seabed_raster_path)
        self.gamma = gamma

    def fit_predict(self, coor, coor_trans, soil_class, **kwargs):
        clf = svm.SVC(kernel='rbf', probability=True, gamma=self.gamma)
        clf.fit(coor_trans, soil_class)
        x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
        y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
        z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1
        vertexCols = np.arange(x_min, x_max, 10)
        vertexRows = np.arange(y_min, y_max, 10)
        vertexLays = np.arange(z_max, z_min, -0.2)
        cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
        cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
        cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
        nCols = cellCols.shape[0]
        nRows = cellRows.shape[0]
        nLays = cellLays.shape[0]
        litoMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
        ProbMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
        lito, ent = self._predict_grid(clf, cellCols, cellRows, cellLays)
        litoMatrix[:, :, :] = lito
        ProbMatrix[:, :, :] = ent
        if self.seabed_raster:
            val, land = self._seabed_masks(cellCols, cellRows)
            # Same equivalence as KNNVoxelModel: land == nodata pixels, which
            # always fail the old "not in land" test; water == valid pixels at
            # or above the seabed elevation.
            water = (~land)[None, :, :] & (cellLays[:, None, None] >= val[None, :, :])
            litoMatrix[water] = 0
            ProbMatrix[water] = ma.masked
        return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix


class GBCVoxelModel(BaseVoxelModel):
    def __init__(self, dist=100, n_estimators=400, seabed_raster_path=None):
        super().__init__(dist, seabed_raster_path)
        self.n_estimators = n_estimators

    def fit_predict(self, coor, coor_trans, soil_class, **kwargs):
        clf = GradientBoostingClassifier(n_estimators=self.n_estimators)
        clf.fit(coor_trans, soil_class)
        x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
        y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
        z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1
        vertexCols = np.arange(x_min, x_max, 50)
        vertexRows = np.arange(y_min, y_max, 50)
        vertexLays = np.arange(z_max, z_min, -1)
        cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
        cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
        cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
        nCols = cellCols.shape[0]
        nRows = cellRows.shape[0]
        nLays = cellLays.shape[0]
        litoMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
        ProbMatrix = ma.array(np.zeros([nLays, nRows, nCols]))
        lito, ent = self._predict_grid(clf, cellCols, cellRows, cellLays)
        litoMatrix[:, :, :] = lito
        ProbMatrix[:, :, :] = ent
        if self.seabed_raster:
            val, land = self._seabed_masks(cellCols, cellRows)
            # GBC additionally masked the entropy of every nodata (land)
            # pixel, at every layer, before the same water rule as KNN/RF/SVM.
            ProbMatrix[np.broadcast_to(land[None, :, :], ProbMatrix.shape)] = ma.masked
            water = (~land)[None, :, :] & (cellLays[:, None, None] >= val[None, :, :])
            litoMatrix[water] = 0
            ProbMatrix[water] = ma.masked
        return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix


class NNVoxelModel(BaseVoxelModel):
    def __init__(self, dist=100, alpha=0.01):
        super().__init__(dist)
        self.alpha = alpha

    def fit_predict(self, coor, coor_trans, soil_class, **kwargs):
        clf = MLPClassifier(activation='logistic', solver='adam', hidden_layer_sizes=(25, 25, 25), max_iter=10000, alpha=self.alpha)
        clf.fit(coor_trans, soil_class)
        x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
        y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
        z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1
        vertexCols = np.arange(x_min, x_max, 1000)
        vertexRows = np.arange(y_min, y_max, 1000)
        vertexLays = np.arange(z_max, z_min, -1)
        cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
        cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
        cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
        nCols = cellCols.shape[0]
        nRows = cellRows.shape[0]
        nLays = cellLays.shape[0]
        litoMatrix = np.zeros([nLays, nRows, nCols])
        ProbMatrix = np.zeros([nLays, nRows, nCols])
        lito, ent = self._predict_grid(clf, cellCols, cellRows, cellLays)
        litoMatrix[:, :, :] = lito
        ProbMatrix[:, :, :] = ent
        return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix


class StackedVoxelModel(BaseVoxelModel):
    def __init__(self, dist=100):
        super().__init__(dist)

    def fit_predict(self, coor, coor_trans, soil_class, **kwargs):
        estimators = [
            ('knn', neighbors.KNeighborsClassifier(15, weights='distance')),
            ('rf', RandomForestClassifier(80))
        ]
        clf = StackingClassifier(estimators=estimators)
        clf.fit(coor_trans, soil_class)
        x_min, x_max = coor[:, 0].min() - 1, coor[:, 0].max() + 1
        y_min, y_max = coor[:, 1].min() - 1, coor[:, 1].max() + 1
        z_min, z_max = coor[:, 2].min() - 1, coor[:, 2].max() + 1
        vertexCols = np.arange(x_min, x_max, 1000)
        vertexRows = np.arange(y_min, y_max, 1000)
        vertexLays = np.arange(z_max, z_min, -1)
        cellCols = (vertexCols[1:] + vertexCols[:-1]) / 2
        cellRows = (vertexRows[1:] + vertexRows[:-1]) / 2
        cellLays = (vertexLays[1:] + vertexLays[:-1]) / 2
        nCols = cellCols.shape[0]
        nRows = cellRows.shape[0]
        nLays = cellLays.shape[0]
        litoMatrix = np.zeros([nLays, nRows, nCols])
        ProbMatrix = np.zeros([nLays, nRows, nCols])
        lito, ent = self._predict_grid(clf, cellCols, cellRows, cellLays)
        litoMatrix[:, :, :] = lito
        ProbMatrix[:, :, :] = ent
        return cellCols, cellRows, cellLays, litoMatrix, ProbMatrix

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified 3D Geological Voxel Modeling")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--out", required=True, help="Output NetCDF file")
    parser.add_argument("--method", required=True, choices=["knn", "rf", "svm", "gbc", "nn", "stacked"], help="Modeling method")
    parser.add_argument("--dist", type=float, default=100, help="Distance scaling factor")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators (RF/GBC)")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for SVM")
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for NN")
    parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors (KNN)")
    parser.add_argument("--seabed", type=str, default=None, help="Seabed raster path (for KNN/RF/SVM/GBC)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.method == "knn":
        model = KNNVoxelModel(dist=args.dist, n_neighbors=args.n_neighbors, seabed_raster_path=args.seabed)
        model.run(df, args.out)
    elif args.method == "rf":
        model = RFVoxelModel(dist=args.dist, n_estimators=args.n_estimators, seabed_raster_path=args.seabed)
        model.run(df, args.out)
    elif args.method == "svm":
        model = SVMVoxelModel(dist=args.dist, gamma=args.gamma, seabed_raster_path=args.seabed)
        model.run(df, args.out)
    elif args.method == "gbc":
        model = GBCVoxelModel(dist=args.dist, n_estimators=args.n_estimators, seabed_raster_path=args.seabed)
        model.run(df, args.out)
    elif args.method == "nn":
        model = NNVoxelModel(dist=args.dist, alpha=args.alpha)
        model.run(df, args.out)
    elif args.method == "stacked":
        model = StackedVoxelModel(dist=args.dist)
        model.run(df, args.out)