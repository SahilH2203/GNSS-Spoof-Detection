from src.feature_engineering import run_feature_engineering
from src.model_training import run_model

def main():
    print("Step 1: Feature Engineering...")
    df_features = run_feature_engineering("data/test.csv")
    
    print("Step 2: Model Prediction...")
    submission = run_model(df_features)
    
    print("Saving submission...")
    submission.to_csv("outputs/submission.csv", index=False)

if __name__ == "__main__":
    main()
