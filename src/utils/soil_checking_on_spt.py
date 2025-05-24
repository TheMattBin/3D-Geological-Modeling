from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import netCDF4 as nc

def load_nc_dataset(nc_path: str) -> nc.Dataset:
    """
    Load a NetCDF dataset from the given path.
    Args:
        nc_path (str): Path to the NetCDF file.
    Returns:
        nc.Dataset: Loaded NetCDF dataset.
    """
    return nc.Dataset(nc_path)

def load_borehole_csv(csv_path: str) -> pd.DataFrame:
    """
    Load borehole data from a CSV file.
    Args:
        csv_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(csv_path)

def prepare_borehole_dicts(df: pd.DataFrame) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[float]]]:
    """
    Prepare dictionaries for elevation, SPTN, and coordinates from DataFrame.
    Args:
        df (pd.DataFrame): Borehole data.
    Returns:
        Tuple of elev, sptn, and coor dicts.
    """
    elev = defaultdict(list)
    sptn = defaultdict(list)
    coor = defaultdict(list)
    l = len(df['Rep_BH_ID'])
    for i in range(l):
        elev[df['Rep_BH_ID'][i]].append([df['TopElev'][i], df['BotElev'][i], df['MidElev'][i]])
        sptn[df['Rep_BH_ID'][i]].append(df['SPTN'][i])
        coor[df['Rep_BH_ID'][i]] = [df['Easting'][i], df['Northing'][i]]
    return elev, sptn, coor

def bh_vs_pred(
    var: nc.Dataset,
    east: float,
    north: float,
    sptn: List[float],
    key: str,
    midelev: List[List[float]],
    predict_list: List[np.ndarray],
    coor_df_east: List[List[float]],
    coor_df_north: List[List[float]],
    key_list: List[List[str]],
    spt_list: List[List[float]]
) -> None:
    """
    Extract predicted soil and entropy along SPT for a borehole location.
    """
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
        midelev.append(Z)
        predict_list.append(predict_bh)
        coor_df_east.append([east] * len(predict_bh))
        coor_df_north.append([north] * len(predict_bh))
        key_list.append([key] * len(predict_bh))
    spt_list.append(sptn)

def predicted_soil_along_spt(
    nc_path: str,
    csv_path: str,
    output_csv: str = "predicted_soil_along_spt.csv"
) -> None:
    """
    Main function to process predicted soil along SPT and save to CSV.
    Args:
        nc_path (str): Path to NetCDF file.
        csv_path (str): Path to borehole CSV file.
        output_csv (str): Output CSV file path.
    """
    var = load_nc_dataset(nc_path)
    df = load_borehole_csv(csv_path)
    elev, sptn, coor = prepare_borehole_dicts(df)
    coor_df_east = []
    coor_df_north = []
    predict_list = []
    spt_list = []
    key_list = []
    midelev = []
    for key in elev.keys():
        bh_vs_pred(
            var,
            coor[key][0],
            coor[key][1],
            sptn[key],
            key,
            midelev,
            predict_list,
            coor_df_east,
            coor_df_north,
            key_list,
            spt_list
        )
    df2 = pd.DataFrame({
        'Location ID': [j for i in key_list for j in i],
        'Easting': [j for i in coor_df_east for j in i],
        'Northing': [j for i in coor_df_north for j in i],
        'MidElev': [j for i in midelev for j in i],
        'Legend Code': [j for i in predict_list for j in i]
    })
    df2.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Example usage; update the paths as needed
    predicted_soil_along_spt(
        nc_path="./data/kNN_airport_250x250x2_100_distance.nc",
        csv_path="./data/airport.csv",
        output_csv="./output/predicted_soil_along_spt.csv"
    )