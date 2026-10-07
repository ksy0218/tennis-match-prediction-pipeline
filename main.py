from preprocessing import load_and_preprocess_data
from model import train_and_evaluate

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df_clean = load_and_preprocess_data()

    print("Training and evaluating model...")
    trained_model = train_and_evaluate(df_clean)

    print("Execution complete.")