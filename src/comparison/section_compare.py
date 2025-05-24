from typing import List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import netCDF4 as nc

import matplotlib.colors
import matplotlib.pyplot as plt


def predicted_bh(
    var: nc.Dataset,
    pt_csv: str,
    output_csv: str
) -> pd.DataFrame:
    """
    Generate predicted borehole lithology and entropy from a 3D model and save to CSV.
    Args:
        var: netCDF4 Dataset for the model.
        pt_csv: Path to CSV file with points along section.
        output_csv: Path to output CSV file.
    Returns:
        DataFrame with predicted lithology and entropy.
    """
    depth_knn = []
    coor_df_east = []
    coor_df_north = []
    predict_list_knn = []
    predict_list_knn_entropy = []
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
        predict_entropy = entropy_matrix[:, idx_north, idx_east]
        if predict_bh.size != 0:
            depth_knn.append(var['z'][:])
            predict_list_knn.append(predict_bh)
            predict_list_knn_entropy.append(predict_entropy)
            coor_df_east.append([east] * len(predict_bh))
            coor_df_north.append([north] * len(predict_bh))

    for e, n in zip(east, north):
        bh_vs_pre(var, e, n)

    topelev = []
    botelev = []

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
    skip_layers: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract lithology and entropy planes for section comparison DataFrame.
    Args:
        df: DataFrame with section comparison data.
        skip_layers: Number of top layers to skip (default 100).
    Returns:
        Tuple of (section_plane, entropy_plane) as numpy arrays.
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


def plot_section_planes(
    section_planes: List[np.ndarray],
    entropy_planes: List[np.ndarray],
    extent: List[float],
    colors: Optional[List[str]] = None
) -> None:
    """
    Plot lithology and entropy planes for multiple models.
    Args:
        section_planes: List of lithology planes (np.ndarray).
        entropy_planes: List of entropy planes (np.ndarray).
        extent: List specifying [xmin, xmax, ymin, ymax] for imshow.
        colors: Optional list of colors for lithology colormap.
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


def main():
    """
    Main entry point for section comparison. Adjust file paths as needed.
    """
    # Example file paths (replace with your own as needed)
    knn_files = [
        'kNN_pc_nohide_10x10x02_100 distance_test.csv',
        'kNN_pc_nohide_10x10x02_1000 distance_test.csv',
        'kNN_pc_nohide_10x10x02_10000 distance_test.csv',
    ]
    section_planes = []
    entropy_planes = []
    for file in knn_files:
        df = pd.read_csv(file)
        section_plane, entropy_plane = xsection_com(df)
        section_planes.append(section_plane)
        entropy_planes.append(entropy_plane)
    extent = [0, 1750, -42, -4]
    plot_section_planes(section_planes, entropy_planes, extent)


if __name__ == "__main__":
    main()

