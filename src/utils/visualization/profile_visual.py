import os
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class GeoProfileVisualizer:
    """
    Class for visualizing borehole and SPT profiles.
    Provides methods to plot borehole lithology profiles and SPT-N profiles.
    """
    @staticmethod
    def load_data(csv_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        Args:
            csv_path (str): Path to the CSV file.
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        return pd.read_csv(csv_path)

    @staticmethod
    def prepare_borehole_depth_and_code(df: pd.DataFrame) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, int]]:
        """
        Prepare depth, code, and color_map dictionaries from DataFrame for borehole profile.
        Args:
            df (pd.DataFrame): Borehole data.
        Returns:
            Tuple of depth dict, code dict, and color_map dict.
        """
        depth = defaultdict(list)
        code = defaultdict(list)
        color_map = {}
        l = len(df['Location ID'])
        for i in range(l):
            depth[df['Location ID'][i]].append(df['Depth Top'][i])
            depth[df['Location ID'][i]].append(df['Depth Base'][i])
            code[df['Location ID'][i]].append(df['Legend Code'][i])
            color_map[df['Geology Code 2'][i]] = df['Legend Code'][i]
        return depth, code, color_map

    @staticmethod
    def plot_borehole_profile(soil: List[int], depth: List[float], color_map: Dict[str, int], key: str, output_dir: str = "./output") -> None:
        """
        Plot and save a borehole profile image.
        Args:
            soil (List[int]): Soil codes for the borehole.
            depth (List[float]): Depth values for the borehole.
            color_map (Dict[str, int]): Mapping from geology code to color index.
            key (str): Borehole identifier.
            output_dir (str): Directory to save the plot.
        """
        os.makedirs(output_dir, exist_ok=True)
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,8))
        cluster = np.repeat(np.expand_dims(soil, 1), 1, 1)
        color_bar = ['white','darkgreen', 'skyblue', 'yellow', 'pink', 'red', 'blue', 'cyan', 'purple', 'orange', 'lightgreen', 'grey']
        cmap_facies = colors.ListedColormap(color_bar[0:len(color_bar)], 'indexed')
        ax.imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies,
                  vmin=1, vmax=len(color_bar), extent=[0,1 ,np.max(depth),np.min(depth)])
        plt.tick_params(bottom=False, labelbottom=False)
        hands = []
        for k in color_map.keys():
            if color_map[k] in soil:
                col = color_bar[(color_map[k])-1]
                hands.append(mpatches.Patch(color=col, label=k))
        plt.legend(handles=hands, loc='best', fontsize=8)
        plt.savefig(os.path.join(output_dir, f"{key}.png"))
        plt.close(f)

    @staticmethod
    def prepare_spt_depth(df: pd.DataFrame) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Prepare depth and SPTN dictionaries from DataFrame for SPT profile.
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

    @staticmethod
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

    @classmethod
    def plot_all_borehole_profiles(cls, csv_path: str, output_dir: str = "./output") -> None:
        """
        Process borehole data and generate lithology profile plots for all boreholes.
        Args:
            csv_path (str): Path to the input CSV file.
            output_dir (str): Directory to save output images.
        """
        df = cls.load_data(csv_path)
        depth, code, color_map = cls.prepare_borehole_depth_and_code(df)
        for key in depth.keys():
            depth_tmp = list(set(depth[key]))
            if '/' in key:
                continue
            cls.plot_borehole_profile(code[key], depth_tmp, color_map, key, output_dir)

    @classmethod
    def plot_all_spt_profiles(cls, csv_path: str, output_dir: str = "./output_spt") -> None:
        """
        Process SPT data and generate SPT-N profile plots for all boreholes.
        Args:
            csv_path (str): Path to the input CSV file.
            output_dir (str): Directory to save output images.
        """
        df = cls.load_data(csv_path)
        depth, SPTN = cls.prepare_spt_depth(df)
        for key in depth.keys():
            spt_tmp = list(SPTN[key])
            dep_tmp = list(depth[key])
            if '/' in key:
                continue
            if len(spt_tmp) >= 2:
                cls.plot_spt_profile(spt_tmp, dep_tmp, key, output_dir)

# Example usage:
if __name__ == "__main__":
    # For borehole lithology profiles:
    # GeoProfileVisualizer:.plot_all_borehole_profiles(csv_path="./data/MHags_V6.csv", output_dir="./output")
    # For SPT profiles:
    # GeoProfileVisualizer:.plot_all_spt_profiles(csv_path="./data/SPT_450.csv", output_dir="./output_spt")
    pass
