import pandas as pd
from typing import List

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

def main():
    # Example usage; update the path as needed
    separate_cpt_sheets(csv_path="./data/CPTcomb.csv")

if __name__ == "__main__":
    main()