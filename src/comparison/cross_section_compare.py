import pandas as pd
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors
from collections import defaultdict, OrderedDict
from typing import List, Optional, Set, Tuple

class SectionComparison:
    """
    Utilities for comparing and visualizing borehole or section plane predictions
    from 3D geological models (netCDF) and ground truth data.
    """

    @staticmethod
    def extract_predicted_borehole(
        ds: nc.Dataset,
        borehole_csv: str,
        output_csv: str
    ) -> pd.DataFrame:
        """
        Generate predicted borehole lithology and entropy from a 3D model and save to CSV.
        """
        depth = defaultdict(list)
        code = defaultdict(list)
        coor = defaultdict(list)
        coor_df_east, coor_df_north = [], []
        predict_list, predict_entropy = [], []
        key_list, soil_list = [], []

        df = pd.read_csv(borehole_csv)
        l = len(df['Location ID'])
        for i in range(l):
            depth[df['Location ID'][i]].append(df['Ground Level'][i] - df['Depth Top'][i])
        for i in range(l):
            depth[df['Location ID'][i]].append(df['Ground Level'][i] - df['Depth Base'][i])
        for i in range(l):
            code[df['Location ID'][i]].append(df['Legend Code'][i])
        for i in range(l):
            coor[df['Location ID'][i]] = ([df['Easting'][i], df['Northing'][i]])

        def bh_vs_pre(var, east, north, key, soil):
            lito_matrix = var['Lithology'][:]
            entropy_matrix = var['Information Entropy'][:]
            X = var['x'][:]
            Y = var['y'][:]
            idx_east = (np.abs(X - east)).argmin()
            idx_north = (np.abs(Y - north)).argmin()
            predict_bh = lito_matrix[:, idx_north, idx_east]
            predict_ent = entropy_matrix[:, idx_north, idx_east]
            if predict_bh.size != 0:
                predict_list.append(predict_bh)
                predict_entropy.append(predict_ent)
                coor_df_east.append([east] * len(predict_bh))
                coor_df_north.append([north] * len(predict_bh))
                key_list.append([key] * len(predict_bh))
            soil_list.append(soil)

        for key in depth.keys():
            mxi = max(depth[key])
            mini = min(depth[key])
            bh_vs_pre(ds, coor[key][0], coor[key][1], key, code[key])

        topelev, botelev = [], []
        z = ds['z'][:]
        for _ in range(len(predict_list)):
            topelev.append(list(z[:] + 0.5))
            tmp = list(z[1:] + 0.5)
            bot = z[-1] - 0.5
            botelev.append(tmp + [bot])

        df2 = pd.DataFrame({
            'Location ID': [j for i in key_list for j in i],
            'Easting': [j for i in coor_df_east for j in i],
            'Northing': [j for i in coor_df_north for j in i],
            'TopElev': [j for i in topelev for j in i],
            'BotElev': [j for i in botelev for j in i],
            'Legend Code': [j for i in predict_list for j in i],
            'Information Entropy': [j for i in predict_entropy for j in i]
        })
        df2['Lithology'] = df2['Legend Code'].map({
            0: 'Water', 1: 'Fill', 2: 'Marine deposit', 3: 'Alluvium', 4: 'Grade V-IV rocks'
        }).fillna('Grade III-I rocks')
        df2['Ground Level'] = ''
        df2['Final Depth'] = ''
        df2.to_csv(output_csv, index=False)
        return df2

    @staticmethod
    def extract_predicted_section(
        ds: nc.Dataset,
        pt_csv: str,
        output_csv: str
    ) -> pd.DataFrame:
        """
        Generate predicted lithology and entropy for a set of section points and save to CSV.
        """
        depth_knn, coor_df_east, coor_df_north = [], [], []
        predict_list, predict_entropy = [], []
        df = pd.read_csv(pt_csv)
        east = list(df['Easting'].values)
        north = list(df['Northing'].values)

        def bh_vs_pre(var, east, north):
            lito_matrix = var['Lithology'][:]
            entropy_matrix = var['Information Entropy'][:]
            X = var['x'][:]
            Y = var['y'][:]
            Z = var['z'][:]
            idx_east = (np.abs(X - east)).argmin()
            idx_north = (np.abs(Y - north)).argmin()
            predict_bh = lito_matrix[:, idx_north, idx_east]
            predict_ent = entropy_matrix[:, idx_north, idx_east]
            if predict_bh.size != 0:
                depth_knn.append(Z)
                predict_list.append(predict_bh)
                predict_entropy.append(predict_ent)
                coor_df_east.append([east] * len(predict_bh))
                coor_df_north.append([north] * len(predict_bh))

        for e, n in zip(east, north):
            bh_vs_pre(ds, e, n)

        topelev, botelev = [], []
        for i in depth_knn:
            topelev.append(list(i[:] + 0.5))
            tmp = list(i[1:] + 0.5)
            bot = i[-1] - 0.5
            botelev.append(tmp + [bot])

        df2 = pd.DataFrame({
            'Easting': [j for i in coor_df_east for j in i],
            'Northing': [j for i in coor_df_north for j in i],
            'TopElev': [j for i in topelev for j in i],
            'BotElev': [j for i in botelev for j in i],
            'Legend Code': [j for i in predict_list for j in i],
            'Information Entropy': [j for i in predict_entropy for j in i]
        })
        df2['Lithology'] = df2['Legend Code'].map({
            0: 'Water', 1: 'Fill', 2: 'Marine deposit', 3: 'Alluvium', 4: 'Grade V-IV rocks'
        }).fillna('Grade III-I rocks')
        df2['Ground Level'] = ''
        df2['Final Depth'] = ''
        df2.to_csv(output_csv, index=False)
        return df2

    @staticmethod
    def extract_section_planes(
        df: pd.DataFrame,
        skip_layers: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract lithology and entropy planes for section comparison DataFrame.
        """
        code = defaultdict(list)
        entropy = defaultdict(list)
        l = len(df['Easting'])
        for i in range(l):
            code[(df['Easting'][i], df['Northing'][i])].append(df['Legend Code'][i])
            entropy[(df['Easting'][i], df['Northing'][i])].append(float(df['Information Entropy'][i]))
        section_plane = [code[key][skip_layers:] for key in code]
        entropy_plane = [entropy[key][skip_layers:] for key in entropy]
        section_plane_trans = np.transpose(np.array(section_plane))
        entropy_plane_trans = np.transpose(np.array(entropy_plane))
        return section_plane_trans, entropy_plane_trans

    @staticmethod
    def plot_section_planes(
        section_planes: List[np.ndarray],
        entropy_planes: List[np.ndarray],
        extent: List[float],
        colors: Optional[List[str]] = None
    ) -> None:
        """
        Plot lithology and entropy planes for multiple models.
        """
        n_models = len(section_planes)
        fig, ax = plt.subplots(nrows=n_models + 1, ncols=1, sharex=True, sharey=True)
        if colors is None:
            colors = ['#002060', '#0070C0', '#66FFFF', '#FFFF00', '#FFC000', '#FF0000']
        cmap = matplotlib.colors.ListedColormap(colors)
        for i, section in enumerate(section_planes):
            ax[i].imshow(section, aspect='auto', cmap=cmap, extent=extent)
        ax[-1].imshow(entropy_planes[1], aspect='auto', cmap=cmap, extent=extent)
        plt.xlabel('Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_entropy_plane_for_boreholes(
        df: pd.DataFrame,
        section_plane_ori: Set[str],
        pt_check_csv: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Visualize entropy plane for a section comparison DataFrame (borehole mode).
        """
        reflist2 = list(section_plane_ori)
        code = defaultdict(list)
        entropy = defaultdict(list)
        east_x = defaultdict(list)
        north_y = defaultdict(list)
        l = len(df['Location ID'])
        for i in range(l):
            if df['Location ID'][i] in reflist2:
                code[df['Location ID'][i]].append(df['Legend Code'][i])
                entropy[df['Location ID'][i]].append(float(df['Information Entropy'][i]))
                east_x[df['Location ID'][i]].append(df['Easting'][i])
                north_y[df['Location ID'][i]].append(df['Northing'][i])
        df_pt = pd.DataFrame({'Easting': [j for i in east_x.values() for j in i], 'Northing': [j for i in north_y.values() for j in i]})
        if pt_check_csv:
            df_pt.to_csv(pt_check_csv, index=False)
        code2 = OrderedDict(sorted(code.items(), key=lambda pair: reflist2.index(pair[0])))
        entropy2 = OrderedDict(sorted(entropy.items(), key=lambda pair: reflist2.index(pair[0])))
        section_plane = [code2[key][20:] for key in code2 if key in section_plane_ori]
        entropy_plane = [entropy2[key][20:] for key in entropy2 if key in section_plane_ori]
        entropy_plane_trans = np.transpose(np.array(entropy_plane))
        en_color = plt.imshow(entropy_plane_trans, aspect='auto', cmap=plt.cm.get_cmap('jet'))
        plt.colorbar(orientation='horizontal')
        plt.colorbar(en_color)
        if show_plot:
            plt.show()
        plt.close()

# Example usage:
if __name__ == "__main__":
    # For borehole prediction extraction:
    # ds = nc.Dataset('your_model.nc')
    # SectionComparison.extract_predicted_borehole(ds, 'borehole.csv', 'predicted_bh.csv')

    # For section prediction extraction:
    # ds = nc.Dataset('your_model.nc')
    # SectionComparison.extract_predicted_section(ds, 'section_points.csv', 'predicted_section.csv')

    # For section plane comparison:
    # knn_files = ['kNN_pc_nohide_10x10x02_100 distance_test.csv', ...]
    # section_planes, entropy_planes = [], []
    # for file in knn_files:
    #     df = pd.read_csv(file)
    #     section_plane, entropy_plane = SectionComparison.extract_section_planes(df)
    #     section_planes.append(section_plane)
    #     entropy_planes.append(entropy_plane)
    # extent = [0, 1750, -42, -4]
    # SectionComparison.plot_section_planes(section_planes, entropy_planes, extent)
    pass