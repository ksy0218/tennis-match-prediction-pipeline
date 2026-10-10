# ATP Match Outcome Predictor (2016–2026)

A machine learning program that forecasts ATP singles match outcomes using historical tour data from 2016-2026. The system features a feature-engineering layer to mitigate target leakage and a gradient boosting classifier.

---

## Technical Highlights

* **Leakage-Free Feature Engineering:** Standard tennis datasets store outcomes via `Winner` and `Loser` columns. To prevent post-match target leakage, a vectorized 50/50 block-swap algorithm randomly assigns players to neutral `Player_A` and `Player_B` representations while generating a balanced binary target vector.
* **Differential Stat Modeling:** Calculates active player rating differentials (`Rank_Diff`, `Pts_Diff`) to capture relative skill gaps prior to tournament rounds.
* **Chronological Time-Series Validation:** Uses sequential train/test splitting (`shuffle=False`) across historical data (2016–2024 train, 2025–2026 test).
* **Native Gradient Boosting Pipeline:** Uses sklearn’s `HistGradientBoostingClassifier` alongside a unified `ColumnTransformer` handling median numeric imputation and categorical one-hot encoding.

---

## Model Performance and Validation

### Gradient Boosting Classifier

<img width="436" height="279" alt="Screenshot 2026-10-09 013113" src="https://github.com/user-attachments/assets/220631c0-0bd6-4929-a81e-4466ad56a387" />

## WIP

Random forest, log regression, web scraping from ATP for player data
