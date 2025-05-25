import pandas as pd
from collections import defaultdict
from typing import Dict, List

class GeoDataCombiner:
    """
    Utility class for combining geotechnical datasets (SPT, CPT, shearbox, etc.).
    """

    @staticmethod
    def combine_geo_spt_shearbox(
        spt_geo_comb_csv: str,
        shb_csv: str,
        output_csv: str = "sptgeo_comb_tri.csv"
    ) -> None:
        """
        Merge SPT-geo combined data with shearbox data and filter by depth.
        """
        spt_geo_comb_df = pd.read_csv(spt_geo_comb_csv)
        shb_data = pd.read_csv(shb_csv)
        df_comb_w_shearbox = spt_geo_comb_df.merge(shb_data, left_on='Hole ID', right_on='HOLE_ID')
        df_comb_w_shearbox = df_comb_w_shearbox[
            (df_comb_w_shearbox['Depth Top'] <= df_comb_w_shearbox['SAMP_TOP']) &
            (df_comb_w_shearbox['SAMP_TOP'] < df_comb_w_shearbox['Depth Base'])
        ]
        df_comb_w_shearbox.to_csv(output_csv, index=False)

    @staticmethod
    def combine_cpt_spt(
        spt_csv: str,
        cpt_csv: str,
        cpt_csv_list: List[str],
        output_csv: str = "combine_cpt_spt.csv"
    ) -> None:
        """
        Combine SPT and CPT data, calculate average Qc for each SPT interval, and merge multiple CPT files.
        """
        sptdata = pd.read_csv(spt_csv)
        sptdata.drop(['BH_Type', 'Report No.', 'Hole ID'], axis=1, inplace=True)
        sptdata.sort_values(by=['Easting_x', 'Northing_x'], inplace=True)
        cptdata = pd.read_csv(cpt_csv)
        cptdata.sort_values(by=['Easting', 'Northing'], inplace=True)
        cpt_elev: Dict = defaultdict(list)
        l = len(cptdata['Easting'])
        spt_elev: Dict = defaultdict(list)
        qc_2_append: Dict = defaultdict(list)
        spt_df = []
        for i in range(l):
            cpt_elev[(cptdata['Easting'][i], cptdata['Northing'][i])].append(cptdata['ElevCPT'][i])
        for key in cpt_elev.keys():
            tmpdf = sptdata[((sptdata['Easting_x']-key[0])**2 + (sptdata['Northing_x']-key[1])**2) <= 12100]
            spt_df.append(tmpdf)
            for i, val in tmpdf.iterrows():
                spt_elev[(val['Easting_x'], val['Northing_x'])].append([val['TopElev_y'], val['BotElev_y']])
        for k, value in spt_elev.items():
            tmpdf = cptdata[((cptdata['Easting'] - k[0]) ** 2 + (cptdata['Northing'] - k[1]) ** 2) <= 12100]
            for v in value:
                tmp_av = tmpdf[(v[1] <= tmpdf['ElevCPT']) & (tmpdf['ElevCPT'] <= v[0])]
                ave_qc = tmp_av['Qc'].mean(axis=0)
                qc_2_append[k].append(ave_qc)
        east = [[k[0]] * len(v) for k, v in qc_2_append.items()]
        north = [[k[1]] * len(v) for k, v in qc_2_append.items()]
        spt_df = pd.concat(spt_df, sort=False)
        spt_df['East'] = [ee for e in east for ee in e]
        spt_df['North'] = [nn for n in north for nn in n]
        spt_df['qc'] = [vv for k, v in qc_2_append.items() for vv in v]
        combdf = pd.concat([pd.read_csv(f) for f in cpt_csv_list], sort=False)
        combdf.to_csv(output_csv, index=False)

class GeoDataSeparator:
    """
    Utility class for separating geotechnical data into Excel files with multiple sheets.
    """
    @staticmethod
    def separate_cpt_sheets(
        csv_path: str,
        output_prefix: str = "output_CPT_",
        group_size: int = 5
    ) -> None:
        """
        Separate CPT data into Excel files, each containing up to group_size sheets.
        Args:
            csv_path (str): Path to the input CSV file.
            output_prefix (str): Prefix for output Excel files.
            group_size (int): Number of sheets per Excel file.
        """
        cptdf = pd.read_csv(csv_path)
        idx: List[int] = [i for i, val in enumerate(cptdf['Check']) if val == 1]
        idx.append(len(cptdf['Check']))
        df_sep = [cptdf.iloc[idx[j]:idx[j+1]].dropna(subset=['STCN_FRES']) for j in range(len(idx)-1)]
        for k in range(0, len(df_sep), group_size):
            with pd.ExcelWriter(f'{output_prefix}{k}.xlsx') as writer:
                for w in range(k, min(k+group_size, len(df_sep))):
                    df_sep[w].to_excel(writer, sheet_name=f'Sheet_name_{w}')

# Example usage:
if __name__ == "__main__":
    combiner = GeoDataCombiner()
    # For geo_spt + shearbox
    # combiner.combine_geo_spt_shearbox(
    #     spt_geo_comb_csv="./data/sptgeo_comb.csv",
    #     shb_csv="./data/Triaxial_Test_Data_V1.csv",
    #     output_csv="sptgeo_comb_tri.csv"
    # )
    # For cpt + spt
    # combiner.combine_cpt_spt(
    #     spt_csv="./data/sptgeo_comb.csv",
    #     cpt_csv="./data/CPT110.csv",
    #     cpt_csv_list=[
    #         "./data/trial_cpt_spt_25.csv",
    #         "./data/trial_cpt_spt_50.csv",
    #         "./data/trial_cpt_spt_80.csv",
    #         "./data/trial_cpt_spt_90.csv",
    #         "./data/trial_cpt_spt_95.csv",
    #         "./data/trial_cpt_spt_100.csv",
    #         "./data/trial_cpt_spt_110.csv"
    #     ],
    #     output_csv="combine_cpt_spt.csv"
    # )
    separator = GeoDataSeparator()
    # For separating CPT sheets
    # separator.separate_cpt_sheets(
    #     csv_path="./data/CPT110.csv",
    #     output_prefix="output_CPT_",
    #     group_size=5
    # )