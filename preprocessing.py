import glob
import numpy as np
import pandas as pd


def load_and_preprocess_data(folder_path='C:/Users/linco/Desktop/tennis/*.xlsx'):
    file_list = glob.glob(folder_path)
    df = pd.concat([pd.read_excel(f) for f in file_list], ignore_index=True)

    #dropping irrelevant and leakage columns
    df = df.drop(df.columns[28:], axis=1)
    cols_to_drop = ['W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 'Wsets', 'Lsets', 'Series', 'Location',
                    'Tournament', 'ATP']
    df = df.drop(cols_to_drop, axis=1)

    #block swap to prevent target leakage
    np.random.seed(42)
    swap = np.random.rand(len(df)) > 0.5

    cols_W = ['Winner', 'WRank', 'WPts']
    cols_L = ['Loser', 'LRank', 'LPts']
    cols_A = ['Player_A', 'Player_A_Rank', 'Player_A_Pts']
    cols_B = ['Player_B', 'Player_B_Rank', 'Player_B_Pts']

    df[cols_A] = df[cols_W]
    df[cols_B] = df[cols_L]
    df.loc[swap, cols_A + cols_B] = df.loc[swap, cols_L + cols_W].values
    df['Player_A_Won'] = (~swap).astype(int)

    #drop remaining leakage columns
    leakage_cols = ['Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts', 'Player_A', 'Player_B', 'Date', 'Comment']
    df_clean = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    #differential features
    df_clean['Rank_Diff'] = df_clean['Player_A_Rank'] - df_clean['Player_B_Rank']
    df_clean['Pts_Diff'] = df_clean['Player_A_Pts'] - df_clean['Player_B_Pts']

    return df_clean