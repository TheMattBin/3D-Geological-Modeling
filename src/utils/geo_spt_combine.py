import pandas as pd

def combine_geo_spt_shearbox(
    spt_geo_comb_csv: str,
    shb_csv: str,
    output_csv: str = "sptgeo_comb_tri.csv"
) -> None:
    """
    Merge SPT-geo combined data with shearbox data and filter by depth.
    Args:
        spt_geo_comb_csv (str): Path to SPT-geo combined CSV file.
        shb_csv (str): Path to shearbox CSV file.
        output_csv (str): Output CSV file path.
    """
    spt_geo_comb_df = pd.read_csv(spt_geo_comb_csv)
    shb_data = pd.read_csv(shb_csv)
    df_comb_w_shearbox = spt_geo_comb_df.merge(shb_data, left_on='Hole ID', right_on='HOLE_ID')
    df_comb_w_shearbox = df_comb_w_shearbox[(df_comb_w_shearbox['Depth Top'] <= df_comb_w_shearbox['SAMP_TOP']) & (df_comb_w_shearbox['SAMP_TOP'] < df_comb_w_shearbox['Depth Base'])]
    df_comb_w_shearbox.to_csv(output_csv, index=False)

def main():
    # Example usage; update the paths as needed
    combine_geo_spt_shearbox(
        spt_geo_comb_csv="./data/sptgeo_comb.csv",
        shb_csv="./data/Triaxial_Test_Data_V1.csv",
        output_csv="sptgeo_comb_tri.csv"
    )

if __name__ == "__main__":
    main()