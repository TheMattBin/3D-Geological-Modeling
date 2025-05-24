import os
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

def load_spt_data(csv_path: str) -> pd.DataFrame:
    """
    Load SPT data from a CSV file.
    Args:
        csv_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(csv_path)

def prepare_spt_depth(df: pd.DataFrame) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Prepare depth and SPTN dictionaries from DataFrame.
    Args:
        df (pd.DataFrame): SPT data.
    Returns:
        Tuple of depth dict and SPTN dict.
    """
    depth = defaultdict(list)
    SPTN = defaultdict(list)
    l = len(df['Location ID'])
    for i in range(l):
        depth[df['Location ID'][i]].append(df['Depth'][i])
        SPTN[df['Location ID'][i]].append(df['N2'][i])
    return depth, SPTN

def plot_spt_profile(spt: List[float], top: List[float], key: str, output_dir: str = "./output_spt") -> None:
    """
    Plot and save an SPT profile image.
    Args:
        spt (List[float]): SPT-N values.
        top (List[float]): Depth values.
        key (str): Borehole identifier.
        output_dir (str): Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.plot(spt, top, 'k-')
    plt.ylabel('Depth (m)')
    plt.xlabel('SPT-N (blows/m)')
    plt.title(f'Depth against SPT-N Graph of {key}')
    plt.axis([0, max(spt), max(top), 0])
    plt.savefig(os.path.join(output_dir, f"{key}.png"))
    plt.clf()

def main(csv_path: str, output_dir: str = "./output_spt") -> None:
    """
    Main function to process SPT data and generate plots.
    Args:
        csv_path (str): Path to the input CSV file.
        output_dir (str): Directory to save output images.
    """
    df = load_spt_data(csv_path)
    depth, SPTN = prepare_spt_depth(df)
    for key in depth.keys():
        spt_tmp = list(SPTN[key])
        dep_tmp = list(depth[key])
        if '/' in key:
            continue
        if len(spt_tmp) >= 2:
            plot_spt_profile(spt_tmp, dep_tmp, key, output_dir)

if __name__ == "__main__":
    # Example usage; update the paths as needed
    main(csv_path="./data/SPT_450.csv", output_dir="./output_spt")