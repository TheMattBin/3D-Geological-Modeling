# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:22:48 2022

@author: Matthew.Liu
"""

import pandas as pd
from typing import Optional

def assign_geocode1(df: pd.DataFrame) -> None:
    # Construction materials
    df.loc[df['GEOL_DESC'].str.contains('FILL|CONCRETE|ASPHALT|BRICK|TILE|Chunam|CEMENT|FOUNDATION WORKS', na=False), 'GeoCode1'] = 'CONSTRUCTION MATERIALS'
    df.loc[df['GEOL_LEG'].str.contains('FILL|CONCRETE|ASPHALT|SURFACE|WALL|MASONRY|LST|BRICK|CEMENT GROUT|PAVIORS|STEEL', na=False), 'GeoCode1'] = 'CONSTRUCTION MATERIALS'
    # Gravel/cobbles/boulder
    df.loc[df['GEOL_DESC'].str.contains('GRAVEL|fragments|COBBLE|BOULDER|COBBLES', na=False), 'GeoCode1'] = 'GRAVEL/COBBLE/BOULDER'
    df.loc[df['GEOL_LEG'].str.contains('GRACBBZS|GRAV|BLDR|CBBL|COBB|GRAB|GRAC', na=False), 'GeoCode1'] = 'GRAVEL/COBBLE/BOULDER'
    # Clay
    df.loc[df['GEOL_DESC'].str.contains('CLAY', na=False), 'GeoCode1'] = 'CLAY'
    df.loc[df['GEOL_LEG'].str.contains('CLAY', na=False), 'GeoCode1'] = 'CLAY'
    # Sand
    df.loc[df['GEOL_DESC'].str.contains('SAND', na=False), 'GeoCode1'] = 'SAND'
    df.loc[df['GEOL_LEG'].str.contains('SAND', na=False), 'GeoCode1'] = 'SAND'
    # Silt
    df.loc[df['GEOL_DESC'].str.contains('SILT', na=False), 'GeoCode1'] = 'SILT'
    df.loc[df['GEOL_LEG'].str.contains('SILT|FINE', na=False), 'GeoCode1'] = 'SILT'
    # Rock types
    df.loc[df['GEOL_LEG'].str.contains('BASALT', na=False), 'GeoCode1'] = 'BASALT'
    df.loc[df['GEOL_LEG'].str.contains('GRANITE', na=False), 'GeoCode1'] = 'GRANITE'
    df.loc[df['GEOL_LEG'].str.contains('GRANODIO|GRANODIORITE', na=False), 'GeoCode1'] = 'GRANODIORITE'
    df.loc[df['GEOL_LEG'].str.contains('RHYOLITE', na=False), 'GeoCode1'] = 'RHYOLITE'
    df.loc[df['GEOL_DESC'].str.contains('METAREG', na=False), 'GeoCode1'] = 'SILTSTONE/SANDSTONE'
    df.loc[df['GEOL_LEG'].str.contains('SILTSTON|SILTSTONE|SANDSTON|SANDSTONE', na=False), 'GeoCode1'] = 'SILTSTONE/SANDSTONE'
    df.loc[df['GEOL_LEG'].str.contains('TUFF|TUFFFINE', na=False), 'GeoCode1'] = 'TUFF'
    # Quartz, other rocks, organics
    df.loc[df['GEOL_DESC'].str.contains('BRECCIA', na=False), 'GeoCode1'] = 'OTHER ROCKS'
    df.loc[df['GEOL_LEG'].str.contains('FAULT|TRACHYTE|GABBRO|MARBLE|CONGLOM|PHYLLITE|METACON|AGGLOM|DACITE|LIMESTONE|MUDSTONE|PEGMITE|SCHIST|SHALE|SYENITE', na=False), 'GeoCode1'] = 'OTHER ROCKS'
    df.loc[df['GEOL_LEG'].str.contains('QUART|VEIN', na=False), 'GeoCode1'] = 'QUARTZOSE ROCKS'
    df.loc[df['GEOL_LEG'].str.contains('BIOCLAST|CORAL|ORGANICS|PEAT', na=False), 'GeoCode1'] = 'MUD/ORGANICS'
    # Blank
    df.loc[df['GEOL_DESC'].str.contains('No sample|Wash boring|No core|WASH BORING', na=False), 'GeoCode1'] = 'N/A'
    df.loc[df['GEOL_LEG'].str.contains('BLANK|WASHING|WB|WASHDRILL|VOID|WASH BORING', na=False), 'GeoCode1'] = 'N/A'

def assign_geocode2(df: pd.DataFrame) -> None:
    # Grade III-I rocks
    df.loc[df['GEOL_DESC'].str.contains('moderately decomposed|slightly decomposed|moderately strong|strong', na=False), 'GeoCode2'] = 'Grade III-I rocks'
    # Grade V-IV rocks
    df.loc[df['GEOL_DESC'].str.contains('completely decomposed|highly decomposed|Very weak|Extremely weak|Moderately weak|weak', na=False), 'GeoCode2'] = 'Grade V-IV rocks'
    # Other deposits/layers
    df.loc[df['GEOL_DESC'].str.contains('BEACH DEPOSIT', na=False), 'GeoCode2'] = 'Beach deposit'
    df.loc[df['GEOL_DESC'].str.contains('COLLUVIUM', na=False), 'GeoCode2'] = 'Colluvium'
    df.loc[df['GEOL_DESC'].str.contains('ALLUVIUM', na=False), 'GeoCode2'] = 'Alluvium'
    df.loc[df['GEOL_DESC'].str.contains('DEBRIS FLOW DEPOSIT', na=False), 'GeoCode2'] = 'Debris flow deposit'
    df.loc[df['GEOL_DESC'].str.contains('ESTUARINE DEPOSIT', na=False), 'GeoCode2'] = 'Estuarine deposit'
    df.loc[df['GEOL_DESC'].str.contains('RESIDUAL SOIL', na=False), 'GeoCode2'] = 'Residual soil'
    df.loc[df['GEOL_DESC'].str.contains('MARINE DEPOSITS|MARINE DEPOSIT|MARINED DEPOSIT|MARINE|Marined|Marine', na=False), 'GeoCode2'] = 'Marine deposit'
    df.loc[df['GEOL_DESC'].str.contains('TOP SOIL|TOPSOIL|MARINED DEPOSIT|FILL|PAVEMENT BRICK|ASPHALT|BRICK|FILTER MATERIAL|GRANITE TILE|ROAD BASE|FOUNDATION WORKS|CEMENT|CONCRETE', na=False), 'GeoCode2'] = 'Fill'
    df.loc[df['GEOL_LEG'].str.contains('FILL|CONCRETE|ASPHALT|SURFACE|WALL|MASONRY|LST|TOPSOIL', na=False), 'GeoCode2'] = 'Fill'
    df.loc[df['GEOL_DESC'].str.contains('CAVITY INFILL|FAULT GOUGE|POND DEPOSIT|LACUSTRINE DEPOSIT|SURFACE KARST|FAULT|KARSTIC DEPOSIT', na=False), 'GeoCode2'] = 'Others'

def fill_missing_geocodes(df: pd.DataFrame) -> None:
    for i, row in df.iterrows():
        if row['GeoCode1'] == 'N/A':
            if ('Wash boring' in str(row['GEOL_DESC']) or 'NO RECOVERY' in str(row['GEOL_DESC']) or 'No sample' in str(row['GEOL_DESC']) or
                'WASH BORING' in str(row['GEOL_DESC']) or 'No core recovery' in str(row['GEOL_DESC']) or 'VOID' in str(row['GEOL_DESC']) or
                'No recovery' in str(row['GEOL_DESC']) or 'Empty boring' in str(row['GEOL_DESC']) or 'Wash Boring' in str(row['GEOL_DESC'])) \
                    and 'completely decomposed' not in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode2'] = 'N/A'
        if pd.isnull(row['GeoCode1']):
            if 'TUFF' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'TUFF'
            elif 'GRANITE' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'GRANITE'
            elif 'BASALT' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'BASALT'
            elif 'GRANODIORITE' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'GRANODIORITE'
            elif 'RHYOLITE' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'RHYOLITE'
            elif 'SANDSTONE' in str(row['GEOL_DESC']) or 'SILTSTONE' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'SILTSTONE/SANDSTONE'
            elif 'QUART' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'QUARTZOSE ROCKS'
            else:
                df.loc[i, 'GeoCode1'] = 'OTHER ROCKS'
        if row['GeoCode2'] == 'Grade III-I rocks':
            if 'TUFF' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'TUFF'
            elif 'GRANITE' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'GRANITE'
            elif 'BASALT' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'BASALT'
            elif 'GRANODIORITE' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'GRANODIORITE'
            elif 'RHYOLITE' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'RHYOLITE'
            elif 'SANDSTONE' in str(row['GEOL_DESC']) or 'SILTSTONE' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'SILTSTONE/SANDSTONE'
            elif 'QUART' in str(row['GEOL_DESC']):
                df.loc[i, 'GeoCode1'] = 'QUARTZOSE ROCKS'
            else:
                df.loc[i, 'GeoCode1'] = 'OTHER ROCKS'
        if pd.isnull(row['GEOL_LEG']) and pd.isnull(row['GEOL_DESC']):
            df.loc[i, 'GeoCode2'] = 'N/A'
            df.loc[i, 'GeoCode1'] = 'N/A'
        if pd.isnull(row['GEOL_DESC']):
            df.loc[i, 'GeoCode2'] = 'N/A'
            df.loc[i, 'GeoCode1'] = 'N/A'

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
    df['Rep_BH_ID'] = 'R_' + df['Report No'] + ' H_' + df['HOLE_ID']
    df.to_csv(output_csv, index=False)

def main():
    # Example usage; update the path as needed
    process_ags_geology(
        input_csv="./data/GEOLcomb.csv",
        output_csv="test2.csv"
    )

if __name__ == "__main__":
    main()