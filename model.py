from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_and_evaluate(df_clean):
    #isolate target
    X = df_clean.drop(columns=['Player_A_Won'])
    y = df_clean['Player_A_Won']

    #train/test split, using 20% of dataset for test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    #colmn definition
    num_cols = ['Best of', 'Player_A_Rank', 'Player_B_Rank', 'Player_A_Pts', 'Player_B_Pts', 'Rank_Diff', 'Pts_Diff']
    cat_cols = ['Court', 'Surface', 'Round']

    # build transformers
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])

    # Build Pipeline and Train
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, predictions))
    print("\n--- ROC-AUC SCORE ---")
    print(f"Score: {roc_auc_score(y_test, probabilities):.3f}")

    return model