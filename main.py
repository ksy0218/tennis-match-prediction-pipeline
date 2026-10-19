from preprocessing import load_and_preprocess_data
from model import train_and_evaluate
from scraper import get_player_stats
import pandas as pd
import joblib

def collect_match_input():
    print("Collecting match input...")
    player_a = input("Enter Player A Name (e.g., Ben Shelton): ").strip()
    player_b = input("Enter Player B Name (e.g., Carlos Alcaraz): ").strip()
    best_of_input = input("Best of (3 or 5): ").strip()
    best_of = int(best_of_input)
    court = input("Court (Outdoor / Indoor): ").strip().capitalize()
    surface = input("Surface (Hard / Clay / Grass): ").strip().capitalize()
    round_name = input("Round (e.g., Finals, Semifinals, R16): ").strip()
    match_data = {
        'Best of': best_of,
        'Player_A_Rank': None,
        'Player_B_Rank': None,
        'Player_A_Pts': None,
        'Player_B_Pts': None,
        'Court': court,
        'Surface': surface,
        'Round': round_name
    }
    return player_a, player_b, match_data

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df_clean = load_and_preprocess_data()
    print(df_clean.head())
    print("Training and evaluating model...")
    trained_model = train_and_evaluate(df_clean)

    print("Execution complete.")
    player_a, player_b, match_data = collect_match_input()
    print(player_a, player_b, match_data)
    
    print(f"\n--- Fetching stats for Player A ({player_a}) ---")
    points_a, rank_a = get_player_stats(player_a)
    match_data['Player_A_Rank'] = rank_a
    match_data['Player_A_Pts'] = points_a

    print(f"\n--- Fetching stats for Player B ({player_b}) ---")
    points_b, rank_b = get_player_stats(player_b)
    match_data['Player_B_Rank'] = rank_b
    match_data['Player_B_Pts'] = points_b

    print("\n=== Final Match Data ===")
    print(match_data)

    match_data['Rank_Diff'] = match_data['Player_A_Rank'] - match_data['Player_B_Rank']
    match_data['Pts_Diff'] = match_data['Player_A_Pts'] - match_data['Player_B_Pts']

    #removing player names from the match data dictionary since the model doesn't need it
    model_features = {k: v for k, v in match_data.items() if 'Name' not in k}
    input_df = pd.DataFrame([model_features])

    pipeline = joblib.load('model.pkl')
    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]

    # 7. Print results
    print("-" * 30)
    if prediction == 1:
        win_prob = probabilities[1] * 100
        print(f"WINNER: {player_a}")
        print(f"Confidence: {win_prob:.1f}%")
    else:
        win_prob = probabilities[0] * 100
        print(f"WINNER: {player_b}")
        print(f"Confidence: {win_prob:.1f}%")
    print("-" * 30)