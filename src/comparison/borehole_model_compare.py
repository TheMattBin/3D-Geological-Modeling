import collections
from collections import defaultdict
from typing import List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc


def predicted_bh(
    ds_svm: nc.Dataset,
    borehole_csv: str,
    output_csv: str
) -> pd.DataFrame:
    """
    Generate predicted borehole lithology and entropy from a 3D model and save to CSV.
    Args:
        ds_svm: netCDF4 Dataset for the model.
        borehole_csv: Path to borehole CSV file.
        output_csv: Path to output CSV file.
    Returns:
        DataFrame with predicted lithology and entropy.
    """
    depth = defaultdict(list)
    code = defaultdict(list)
    coor = defaultdict(list)
    coor_df_east = []
    coor_df_north = []
    predict_list_knn = []
    soil_list = []
    key_list = []
    predict_list_knn_entropy = []

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

    def bh_vs_pre(
        var: nc.Dataset, east: float, north: float, depth: float, depth_t: float, depth_b: float, soil: List[Any], key: str
    ):
        lito_matrix = var['Lithology'][:]
        entropy_matrix = var['Information Entropy'][:]
        X = var['x'][:]
        Y = var['y'][:]
        Z = var['z'][:]
        idx_depth = (np.abs(Z - depth)).argmin()
        idx_depth_top = (np.abs(Z - depth_t)).argmin()
        idx_depth_bot = (np.abs(Z - depth_b)).argmin()
        idx_east = (np.abs(X - east)).argmin()
        idx_north = (np.abs(Y - north)).argmin()
        predict_bh = lito_matrix[:, idx_north, idx_east]
        predict_entropy = entropy_matrix[:, idx_north, idx_east]
        if predict_bh.size != 0:
            depth_knn = var['z'][:]
            predict_list_knn.append(predict_bh)
            predict_list_knn_entropy.append(predict_entropy)
            coor_df_east.append([east] * len(predict_bh))
            coor_df_north.append([north] * len(predict_bh))
            key_list.append([key] * len(predict_bh))
        soil_list.append(soil)

    for key in depth.keys():
        mxi = max(depth[key])
        mini = min(depth[key])
        bh_vs_pre(ds_svm, coor[key][0], coor[key][1], mxi, mxi, mini, code[key], key)

    topelev = []
    botelev = []
    for i in range(len(predict_list_knn)):
        z = ds_svm['z'][:]
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
        'Legend Code': [j for i in predict_list_knn for j in i],
        'Information Entropy': [j for i in predict_list_knn_entropy for j in i]
    })
    df2['Lithology'] = df2['Legend Code'].map({
        0: 'Water', 1: 'Fill', 2: 'Marine deposit', 3: 'Alluvium', 4: 'Grade V-IV rocks'
    }).fillna('Grade III-I rocks')
    df2['Ground Level'] = ''
    df2['Final Depth'] = ''
    df2.to_csv(output_csv, index=False)
    return df2

def xsection_com(
    df: pd.DataFrame,
    section_plane_ori: set,
    pt_check_csv: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Visualize entropy plane for a section comparison DataFrame.
    Args:
        df: DataFrame with section comparison data.
        section_plane_ori: Set of location IDs for section plane.
        pt_check_csv: Optional path to save point check CSV.
        show_plot: Whether to display the plot.
    """
    reflist2 = [
        'R_3572 H_MC36', 'R_38469 H_MBH2', 'R_2412 H_PC301', 'R_37353 H_DH1', 'R_25765 H_TL 1',
        'R_37353 H_DH2', 'R_25765 H_TL 3', 'R_25765 H_TL 2', 'R_37353 H_DH3', 'R_37353 H_DH6',
        'R_2412 H_PC303', 'R_37353 H_DH5', 'R_2412 H_PC304', 'R_2412 H_PC305', 'R_2412 H_PC307', 'R_2412 H_PC309',
        'R_2412 H_PC310', 'R_2412 H_PC314', 'R_2412 H_PC313', 'R_2412 H_PC315', 'R_2412 H_PC316', 'R_49001 H_BH3', 'R_49001 H_BH4',
        'R_2412 H_PC317', 'R_49001 H_BH2', 'R_49001 H_BH1', 'R_2412 H_PC318', 'R_25940 H_MS14', 'R_2412 H_PC319',
        'R_25940 H_MS13', 'R_25940 H_BH16', 'R_25940 H_MS12', 'R_2412 H_PC320', 'R_25940 H_MS11',
        'R_25940 H_BH15', 'R_25940 H_MS10', 'R_25940 H_MS9', 'R_2412 H_PC322', 'R_25940 H_BH13', 'R_25940 H_BH12',
        'R_2412 H_PC323', 'R_2412 H_PC324', 'R_32389 H_PS 2', 'R_2412 H_PC325', 'R_32389 H_PS 1', 'R_2412 H_PC326',
        'R_2412 H_PC327', 'R_2412 H_PC328', 'R_2412 H_PC329', 'R_2412 H_PC330', 'R_2412 H_PC331', 'R_2412 H_PC332',
        'R_2412 H_PC333', 'R_2412 H_PC334', 'R_2412 H_PC335', 'R_2412 H_PC336', 'R_2412 H_PC337', 'R_2412 H_PC338',
        'R_2412 H_PC339', 'R_32005 H_B4', 'R_32005 H_B6', 'R_32654 H_B 2', 'R_32654 H_B 1', 'R_29358 H_DH 4', 'R_32654 H_B 5',
        'R_32654 H_B 3', 'R_32654 H_B 7', 'R_42887 H_BH-10', 'R_42887 H_BH-11', 'R_42887 H_BH-09',
        'R_20517 H_BH-08', 'R_42887 H_BH-01', 'R_42887 H_BH-02', 'R_42887 H_BH-03', 'R_42887 H_BH-04',
        'R_20501 H_BH6', 'R_20501 H_BH7'
    ]
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
    code2 = collections.OrderedDict(sorted(code.items(), key=lambda pair: reflist2.index(pair[0])))
    entropy2 = collections.OrderedDict(sorted(entropy.items(), key=lambda pair: reflist2.index(pair[0])))
    section_plane = [code2[key][20:] for key in code2 if key in section_plane_ori]
    entropy_plane = [entropy2[key][20:] for key in entropy2 if key in section_plane_ori]
    entropy_plane_trans = np.transpose(np.array(entropy_plane))
    en_color = plt.imshow(entropy_plane_trans, aspect='auto', cmap=plt.cm.get_cmap('jet'))
    plt.colorbar(orientation='horizontal')
    plt.colorbar(en_color)
    if show_plot:
        plt.show()
    plt.close()

def main():
    """
    Main entry point for borehole model comparison. Adjust file paths as needed.
    """
    # Example file paths (replace with your own as needed)
    section_compare_csv = 'Section_Compare.csv'
    knn100_pc_nh_csv = 'kNN_pc_nohide_50x50x1_100 distance_test.csv'
    pt_check_csv = 'pt_check.csv'
    # Load section plane original set
    df0 = pd.read_csv(section_compare_csv)
    section_plane_ori = set(df0['Location ID'])
    # Load DataFrame for comparison
    df100_pc_nh = pd.read_csv(knn100_pc_nh_csv)
    # Run section comparison visualization
    xsection_com(df100_pc_nh, section_plane_ori, pt_check_csv=pt_check_csv)

if __name__ == "__main__":
    main()