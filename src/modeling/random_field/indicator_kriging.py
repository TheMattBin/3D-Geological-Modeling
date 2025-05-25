import numpy as np
import pandas as pd

# Parameters
EXCEL_FILE = 'CentralWater_20211230.csv'
SOFH = 100
SOFV = 0.0001
NUM_LAYERS = 12

def CorrFun(dx, dy, dz, SOFH, SOFV):
    return np.exp(-2 * np.sqrt(dx ** 2 + dy ** 2) / SOFH - 2 * dz / SOFV)

def assign_indicator_probabilities(df, num_layers):
    # Assign 1 to the corresponding indicator column based on 'Legend Code'
    for i in range(num_layers):
        col = f'P(I={i})'
        df[col] = (df['Legend Code'] == i).astype(int)
    return df

def get_indicator_matrix(df, num_layers):
    # Fill NaNs with 0 and extract indicator columns as numpy array
    indicator_cols = [f'P(I={i})' for i in range(num_layers)]
    df[indicator_cols] = df[indicator_cols].fillna(0)
    I = df[indicator_cols].values
    return I

def main():
    df = pd.read_csv(EXCEL_FILE)
    df = assign_indicator_probabilities(df, NUM_LAYERS)

    # Compute mean probability for each indicator
    indicator_cols = [f'P(I={i})' for i in range(NUM_LAYERS)]
    prob_mean = df[indicator_cols].mean(axis=0)
    Im = prob_mean.values

    # Transform into indicator matrix and center by mean
    I = get_indicator_matrix(df, NUM_LAYERS)
    I = I - Im

    # Coordinates
    coordinates = df[['Easting', 'Northing', 'ElevMid']].values
    X = coordinates[:, 0][:, np.newaxis]
    Y = coordinates[:, 1][:, np.newaxis]
    Z = coordinates[:, 2][:, np.newaxis]

    # Compute correlation matrix for known points
    x1, x2 = np.meshgrid(X, X)
    y1, y2 = np.meshgrid(Y, Y)
    z1, z2 = np.meshgrid(Z, Z)
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)
    dz = np.abs(z1 - z2)
    Rkk = CorrFun(dx, dy, dz, SOFH, SOFV)

    # Create grid for 3D model
    x_min, x_max = X.min() - 1, X.max() + 1
    y_min, y_max = Y.min() - 1, Y.max() + 1
    z_min, z_max = Z.min() - 1, Z.max() + 1
    x = np.linspace(x_min, x_max, 20)
    y = np.linspace(y_min, y_max, 20)
    z = np.linspace(z_max, z_min, 10)
    Zu, Yu, Xu = np.meshgrid(z, y, x, indexing='ij')

    # Correlation between unknown and known locations
    x1u, x2u = np.meshgrid(X, Xu)
    y1u, y2u = np.meshgrid(Y, Yu)
    z1u, z2u = np.meshgrid(Z, Zu)
    dxu = np.abs(x1u - x2u)
    dyu = np.abs(y1u - y2u)
    dzu = np.abs(z1u - z2u)
    Ruk = CorrFun(dxu, dyu, dzu, SOFH, SOFV)

    # Kriging prediction
    Rkkinv = np.linalg.inv(Rkk)
    Predict = np.matmul(np.matmul(Ruk, Rkkinv), I) + Im

    # Optionally, return or save Predict here

if __name__ == "__main__":
    main()