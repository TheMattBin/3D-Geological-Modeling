# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:22:48 2022

@author: Matthew.Liu
"""

import pandas as pd

def cleanse_borehole_data(
    input_csv: str,
    output_csv: str = 'test5.csv'
) -> None:
    """
    Cleanse borehole data, remove duplicates, fill missing, check errors, and assign legend codes.
    """
    df = pd.read_csv(input_csv)
    df.drop_duplicates(inplace=True)
    df.fillna(value="N/A", inplace=True)
    uid = df['Rep_BH_ID'].unique()
    df.set_index("Rep_BH_ID", inplace=True)
    for i in uid:
        a = df.loc[i]
        if len(a.shape) >= 2:
            t = a['TopDepth']
            b = a['BotDepth'].shift()
            if (b-t)[1:].any() != 0:
                df.loc[i, 'error'] = 1
            if a['BotDepth'].values[-1] != a['Depth'].values[-1]:
                df.loc[i, 'error'] = 1
    # Legend Code
    df.loc[df['GeoCode2'].str.contains('N/A', na=False), 'Code'] = 0
    df.loc[df['GeoCode2'].str.contains('Fill', na=False), 'Code'] = 1
    df.loc[df['GeoCode2'].str.contains('Beach deposit', na=False), 'Code'] = 2
    df.loc[df['GeoCode2'].str.contains('Marine deposit', na=False), 'Code'] = 3
    df.loc[df['GeoCode2'].str.contains('Estuarine deposit', na=False), 'Code'] = 4
    df.loc[df['GeoCode2'].str.contains('Alluvium', na=False), 'Code'] = 5
    df.loc[df['GeoCode2'].str.contains('Debris flow deposit', na=False), 'Code'] = 6
    df.loc[df['GeoCode2'].str.contains('Colluvium', na=False), 'Code'] = 7
    df.loc[df['GeoCode2'].str.contains('Residual soil', na=False), 'Code'] = 8
    df.loc[df['GeoCode2'].str.contains('Grade V-IV rocks', na=False), 'Code'] = 9
    df.loc[df['GeoCode2'].str.contains('Grade III-I rocks', na=False), 'Code'] = 10
    df.loc[df['GeoCode2'].str.contains('Others', na=False), 'Code'] = 11
    df.to_csv(output_csv)

def main():
    # Example usage; update the path as needed
    cleanse_borehole_data(
        input_csv='BH_20250962.csv',
        output_csv='test5.csv'
    )

if __name__ == "__main__":
    main()