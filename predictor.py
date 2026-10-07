import glob
import numpy as np
import pandas as pd
file_list = glob.glob('C:/Users/linco/Desktop/tennis/*.xlsx')
for f in file_list:
    print(f)
df = pd.concat([pd.read_excel(f) for f in file_list], ignore_index=True)


print(df.columns[28:])
df = df.drop(df.columns[28:], axis=1) #removing columns related to betting which won't be used in this project
cols_to_drop = ['W1','L1','W2','L2','W3','L3','W4','L4','W5','L5','Wsets','Lsets','Series','Location','Tournament','ATP']
df = df.drop(cols_to_drop, axis=1) #data leakage or irrelevant columns
#sorting the dataframe chronologically
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)


np.random.seed(42)
swap = np.random.rand(len(df)) > 0.5

# Group all Winner columns and Loser columns
cols_W = ['Winner', 'WRank', 'WPts']
cols_L = ['Loser', 'LRank', 'LPts']

# Define the new neutral column names
cols_A = ['Player_A', 'Player_A_Rank', 'Player_A_Pts']
cols_B = ['Player_B', 'Player_B_Rank', 'Player_B_Pts']

# Temporarily assign A to Winner and B to Loser for all rows
df[cols_A] = df[cols_W]
df[cols_B] = df[cols_L]

# For the ~50% of rows where 'swap' is True, flip A and B simultaneously in one line
df.loc[swap, cols_A + cols_B] = df.loc[swap, cols_L + cols_W].values

# Create the target variable: 1 if we didn't swap (A won), 0 if we swapped (A lost)
# Using ~swap inverts True to False, and .astype(int) turns it into 1s and 0s
df['Player_A_Won'] = (~swap).astype(int)


#code
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Eradicate Leakage Columns
# Dropping the original columns, the raw names, and 'Date'/'Comment' if they exist
leakage_cols = ['Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts', 'Player_A', 'Player_B', 'Date', 'Comment']
df_clean = df.drop(columns=[c for c in leakage_cols if c in df.columns])

# 2. Engineer Differential Features
# This explicitly calculates the skill gap, drastically speeding up the model's learning
df_clean['Rank_Diff'] = df_clean['Player_A_Rank'] - df_clean['Player_B_Rank']
df_clean['Pts_Diff'] = df_clean['Player_A_Pts'] - df_clean['Player_B_Pts']

# 3. Isolate Features (X) and Target (y)
X = df_clean.drop(columns=['Player_A_Won'])
y = df_clean['Player_A_Won']

# 4. Chronological Train/Test Split
# shuffle=False guarantees we train on historical data (e.g., 2016-2024) to predict recent data (2025-2026)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. Define Column Types for Processing
num_cols = ['Best of', 'Player_A_Rank', 'Player_B_Rank', 'Player_A_Pts', 'Player_B_Pts', 'Rank_Diff', 'Pts_Diff']
cat_cols = ['Court', 'Surface', 'Round']

# Define the rules: Impute missing numbers with the median, then scale them
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define the rules: Impute missing text with the most frequent category, then One-Hot Encode
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Bundle the rules into the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# 6. Build the Final Pipeline & Train
# HistGradientBoostingClassifier is used here as it performs similarly to XGBoost/LightGBM
# and is extremely efficient for datasets with complex interactions.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# Pushes the raw X_train data through the imputation, encoding, scaling, and training automatically
model.fit(X_train, y_train)

# 7. Validate the Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, predictions))

print("\n--- ROC-AUC SCORE ---")
print(f"Score: {roc_auc_score(y_test, probabilities):.3f}")