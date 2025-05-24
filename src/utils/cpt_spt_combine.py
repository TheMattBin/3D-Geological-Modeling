import pandas as pd
from collections import defaultdict
from typing import Dict, List


def combine_cpt_spt(
    spt_csv: str,
    cpt_csv: str,
    cpt_csv_list: List[str],
    output_csv: str = "combine_cpt_spt.csv"
) -> None:
    """
    Combine SPT and CPT data, calculate average Qc for each SPT interval, and merge multiple CPT files.
    Args:
        spt_csv (str): Path to SPT CSV file.
        cpt_csv (str): Path to main CPT CSV file.
        cpt_csv_list (List[str]): List of CPT CSV files to concatenate.
        output_csv (str): Output CSV file path.
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

    dfqc = pd.DataFrame(
        {'Easting': [ee for e in east for ee in e], 'Northing': [nn for n in north for nn in n],
         'qc': [vv for k, v in qc_2_append.items() for vv in v]})
    spt_df = pd.concat(spt_df, sort=False)
    spt_df['East'] = [ee for e in east for ee in e]
    spt_df['North'] = [nn for n in north for nn in n]
    spt_df['qc'] = [vv for k, v in qc_2_append.items() for vv in v]
    combdf = pd.concat([pd.read_csv(f) for f in cpt_csv_list], sort=False)
    combdf.to_csv(output_csv, index=False)


def main():
    # Example usage; update the paths as needed
    combine_cpt_spt(
        spt_csv="./data/sptgeo_comb.csv",
        cpt_csv="./data/CPT110.csv",
        cpt_csv_list=[
            "./data/trial_cpt_spt_25.csv",
            "./data/trial_cpt_spt_50.csv",
            "./data/trial_cpt_spt_80.csv",
            "./data/trial_cpt_spt_90.csv",
            "./data/trial_cpt_spt_95.csv",
            "./data/trial_cpt_spt_100.csv",
            "./data/trial_cpt_spt_110.csv"
        ],
        output_csv="combine_cpt_spt.csv"
    )


if __name__ == "__main__":
    main()