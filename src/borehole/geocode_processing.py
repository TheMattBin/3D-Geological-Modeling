# -*- coding: utf-8 -*-
"""
AGS Geology CSV Processing Script

This script processes AGS (Association of Geotechnical and Geoenvironmental Specialists) geology CSV files.
It assigns standardized geological codes (GeoCode1 and GeoCode2) to each record based on descriptive fields,
cleans the data, and outputs a processed CSV file. The main steps are:
- Assign GeoCode1 and GeoCode2 based on geological descriptions and legends
- Fill missing or ambiguous codes using additional logic
- Output a cleaned CSV with new codes and a unique borehole ID

Author: Matthew.Liu
Refactored: May 2025
"""

import pandas as pd

def assign_geocode1(df: pd.DataFrame) -> None:
    """
    Assign GeoCode1 based on geological description and legend fields.
    """
    # Construction materials
    df.loc[df['GEOL_DESC'].str.contains(r'FILL|CONCRETE|ASPHALT|BRICK|TILE|Chunam|CEMENT|FOUNDATION WORKS', na=False, case=False), 'GeoCode1'] = 'CONSTRUCTION MATERIALS'
    df.loc[df['GEOL_LEG'].str.contains(r'FILL|CONCRETE|ASPHALT|SURFACE|WALL|MASONRY|LST|BRICK|CEMENT GROUT|PAVIORS|STEEL', na=False, case=False), 'GeoCode1'] = 'CONSTRUCTION MATERIALS'
    # Gravel/cobbles/boulder
    df.loc[df['GEOL_DESC'].str.contains(r'GRAVEL|fragments|COBBLE|BOULDER|COBBLES', na=False, case=False), 'GeoCode1'] = 'GRAVEL/COBBLE/BOULDER'
    df.loc[df['GEOL_LEG'].str.contains(r'GRACBBZS|GRAV|BLDR|CBBL|COBB|GRAB|GRAC', na=False, case=False), 'GeoCode1'] = 'GRAVEL/COBBLE/BOULDER'
    # Clay
    df.loc[df['GEOL_DESC'].str.contains(r'CLAY', na=False, case=False), 'GeoCode1'] = 'CLAY'
    df.loc[df['GEOL_LEG'].str.contains(r'CLAY', na=False, case=False), 'GeoCode1'] = 'CLAY'
    # Sand
    df.loc[df['GEOL_DESC'].str.contains(r'SAND', na=False, case=False), 'GeoCode1'] = 'SAND'
    df.loc[df['GEOL_LEG'].str.contains(r'SAND', na=False, case=False), 'GeoCode1'] = 'SAND'
    # Silt
    df.loc[df['GEOL_DESC'].str.contains(r'SILT', na=False, case=False), 'GeoCode1'] = 'SILT'
    df.loc[df['GEOL_LEG'].str.contains(r'SILT|FINE', na=False, case=False), 'GeoCode1'] = 'SILT'
    # Rock types
    df.loc[df['GEOL_LEG'].str.contains(r'BASALT', na=False, case=False), 'GeoCode1'] = 'BASALT'
    df.loc[df['GEOL_LEG'].str.contains(r'GRANITE', na=False, case=False), 'GeoCode1'] = 'GRANITE'
    df.loc[df['GEOL_LEG'].str.contains(r'GRANODIO|GRANODIORITE', na=False, case=False), 'GeoCode1'] = 'GRANODIORITE'
    df.loc[df['GEOL_LEG'].str.contains(r'RHYOLITE', na=False, case=False), 'GeoCode1'] = 'RHYOLITE'
    df.loc[df['GEOL_DESC'].str.contains(r'METAREG', na=False, case=False), 'GeoCode1'] = 'SILTSTONE/SANDSTONE'
    df.loc[df['GEOL_LEG'].str.contains(r'SILTSTON|SILTSTONE|SANDSTON|SANDSTONE', na=False, case=False), 'GeoCode1'] = 'SILTSTONE/SANDSTONE'
    df.loc[df['GEOL_LEG'].str.contains(r'TUFF|TUFFFINE', na=False, case=False), 'GeoCode1'] = 'TUFF'
    # Quartz, other rocks, organics
    df.loc[df['GEOL_DESC'].str.contains(r'BRECCIA', na=False, case=False), 'GeoCode1'] = 'OTHER ROCKS'
    df.loc[df['GEOL_LEG'].str.contains(r'FAULT|TRACHYTE|GABBRO|MARBLE|CONGLOM|PHYLLITE|METACON|AGGLOM|DACITE|LIMESTONE|MUDSTONE|PEGMITE|SCHIST|SHALE|SYENITE', na=False, case=False), 'GeoCode1'] = 'OTHER ROCKS'
    df.loc[df['GEOL_LEG'].str.contains(r'QUART|VEIN', na=False, case=False), 'GeoCode1'] = 'QUARTZOSE ROCKS'
    df.loc[df['GEOL_LEG'].str.contains(r'BIOCLAST|CORAL|ORGANICS|PEAT', na=False, case=False), 'GeoCode1'] = 'MUD/ORGANICS'
    # Blank
    df.loc[df['GEOL_DESC'].str.contains(r'No sample|Wash boring|No core|WASH BORING', na=False, case=False), 'GeoCode1'] = 'N/A'
    df.loc[df['GEOL_LEG'].str.contains(r'BLANK|WASHING|WB|WASHDRILL|VOID|WASH BORING', na=False, case=False), 'GeoCode1'] = 'N/A'

def assign_geocode2(df: pd.DataFrame) -> None:
    """
    Assign GeoCode2 based on geological description and legend fields.
    """
    # Grade III-I rocks
    df.loc[df['GEOL_DESC'].str.contains(r'moderately decomposed|slightly decomposed|moderately strong|strong', na=False, case=False), 'GeoCode2'] = 'Grade III-I rocks'
    # Grade V-IV rocks
    df.loc[df['GEOL_DESC'].str.contains(r'completely decomposed|highly decomposed|Very weak|Extremely weak|Moderately weak|weak', na=False, case=False), 'GeoCode2'] = 'Grade V-IV rocks'
    # Other deposits/layers
    df.loc[df['GEOL_DESC'].str.contains(r'BEACH DEPOSIT', na=False, case=False), 'GeoCode2'] = 'Beach deposit'
    df.loc[df['GEOL_DESC'].str.contains(r'COLLUVIUM', na=False, case=False), 'GeoCode2'] = 'Colluvium'
    df.loc[df['GEOL_DESC'].str.contains(r'ALLUVIUM', na=False, case=False), 'GeoCode2'] = 'Alluvium'
    df.loc[df['GEOL_DESC'].str.contains(r'DEBRIS FLOW DEPOSIT', na=False, case=False), 'GeoCode2'] = 'Debris flow deposit'
    df.loc[df['GEOL_DESC'].str.contains(r'ESTUARINE DEPOSIT', na=False, case=False), 'GeoCode2'] = 'Estuarine deposit'
    df.loc[df['GEOL_DESC'].str.contains(r'RESIDUAL SOIL', na=False, case=False), 'GeoCode2'] = 'Residual soil'
    df.loc[df['GEOL_DESC'].str.contains(r'MARINE DEPOSITS|MARINE DEPOSIT|MARINED DEPOSIT|MARINE|Marined|Marine', na=False, case=False), 'GeoCode2'] = 'Marine deposit'
    df.loc[df['GEOL_DESC'].str.contains(r'TOP SOIL|TOPSOIL|MARINED DEPOSIT|FILL|PAVEMENT BRICK|ASPHALT|BRICK|FILTER MATERIAL|GRANITE TILE|ROAD BASE|FOUNDATION WORKS|CEMENT|CONCRETE', na=False, case=False), 'GeoCode2'] = 'Fill'
    df.loc[df['GEOL_LEG'].str.contains(r'FILL|CONCRETE|ASPHALT|SURFACE|WALL|MASONRY|LST|TOPSOIL', na=False, case=False), 'GeoCode2'] = 'Fill'
    df.loc[df['GEOL_DESC'].str.contains(r'CAVITY INFILL|FAULT GOUGE|POND DEPOSIT|LACUSTRINE DEPOSIT|SURFACE KARST|FAULT|KARSTIC DEPOSIT', na=False, case=False), 'GeoCode2'] = 'Others'

def fill_missing_geocodes(df: pd.DataFrame) -> None:
    """
    Fill missing or ambiguous GeoCode1/GeoCode2 values using additional logic.
    """
    for i, row in df.iterrows():
        # Set GeoCode2 to N/A for certain N/A GeoCode1 rows
        if row.get('GeoCode1') == 'N/A':
            desc = str(row.get('GEOL_DESC', ''))
            if (any(x in desc for x in ['Wash boring', 'NO RECOVERY', 'No sample', 'WASH BORING', 'No core recovery', 'VOID', 'No recovery', 'Empty boring', 'Wash Boring'])
                and 'completely decomposed' not in desc):
                df.at[i, 'GeoCode2'] = 'N/A'
        # Fill missing GeoCode1 based on description
        if pd.isnull(row.get('GeoCode1')):
            desc = str(row.get('GEOL_DESC', ''))
            if 'TUFF' in desc:
                df.at[i, 'GeoCode1'] = 'TUFF'
            elif 'GRANITE' in desc:
                df.at[i, 'GeoCode1'] = 'GRANITE'
            elif 'BASALT' in desc:
                df.at[i, 'GeoCode1'] = 'BASALT'
            elif 'GRANODIORITE' in desc:
                df.at[i, 'GeoCode1'] = 'GRANODIORITE'
            elif 'RHYOLITE' in desc:
                df.at[i, 'GeoCode1'] = 'RHYOLITE'
            elif 'SANDSTONE' in desc or 'SILTSTONE' in desc:
                df.at[i, 'GeoCode1'] = 'SILTSTONE/SANDSTONE'
            elif 'QUART' in desc:
                df.at[i, 'GeoCode1'] = 'QUARTZOSE ROCKS'
            else:
                df.at[i, 'GeoCode1'] = 'OTHER ROCKS'
        # If GeoCode2 is Grade III-I rocks, update GeoCode1
        if row.get('GeoCode2') == 'Grade III-I rocks':
            desc = str(row.get('GEOL_DESC', ''))
            if 'TUFF' in desc:
                df.at[i, 'GeoCode1'] = 'TUFF'
            elif 'GRANITE' in desc:
                df.at[i, 'GeoCode1'] = 'GRANITE'
            elif 'BASALT' in desc:
                df.at[i, 'GeoCode1'] = 'BASALT'
            elif 'GRANODIORITE' in desc:
                df.at[i, 'GeoCode1'] = 'GRANODIORITE'
            elif 'RHYOLITE' in desc:
                df.at[i, 'GeoCode1'] = 'RHYOLITE'
            elif 'SANDSTONE' in desc or 'SILTSTONE' in desc:
                df.at[i, 'GeoCode1'] = 'SILTSTONE/SANDSTONE'
            elif 'QUART' in desc:
                df.at[i, 'GeoCode1'] = 'QUARTZOSE ROCKS'
            else:
                df.at[i, 'GeoCode1'] = 'OTHER ROCKS'
        # If both GEOL_LEG and GEOL_DESC are missing, set both codes to N/A
        if pd.isnull(row.get('GEOL_LEG')) and pd.isnull(row.get('GEOL_DESC')):
            df.at[i, 'GeoCode2'] = 'N/A'
            df.at[i, 'GeoCode1'] = 'N/A'
        # If GEOL_DESC is missing, set both codes to N/A
        if pd.isnull(row.get('GEOL_DESC')):
            df.at[i, 'GeoCode2'] = 'N/A'
            df.at[i, 'GeoCode1'] = 'N/A'

def process_ags_geology(
    input_csv: str,
    output_csv: str = 'test2.csv'
) -> None:
    """
    Process AGS geology CSV, assign GeoCode1 and GeoCode2, and output cleaned CSV.
    """
    df = pd.read_csv(input_csv)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['HOLE_NATE', 'HOLE_NATN', 'HOLE_GL', 'HOLE_FDEP', 'GEOL_TOP', 'GEOL_BASE'], inplace=True)
    assign_geocode1(df)
    assign_geocode2(df)
    fill_missing_geocodes(df)
    df.fillna(value="N/A", inplace=True)
    df['Rep_BH_ID'] = 'R_' + df['Report No'].astype(str) + ' H_' + df['HOLE_ID'].astype(str)
    df.to_csv(output_csv, index=False)

def main():
    """
    Example usage; update the path as needed.
    """
    process_ags_geology(
        input_csv="./data/GEOLcomb.csv",
        output_csv="test2.csv"
    )

if __name__ == "__main__":
    main()