import pandas as pd

from sklearn.model_selection import train_test_split

from preprocessing import preprocess_data
from evaluate import evaluate_model
from models import train_logistic_regression, train_random_forest
from compare import get_model_metrics, compare_models

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nLogistic Regression")
    log_model, _, X_test_scaled = train_logistic_regression(
        X_train,
        y_train,
        X_test
    )
    evaluate_model(log_model, X_test_scaled, y_test)

    print("\nRandom Forest")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    results = [
        get_model_metrics(
            log_model,
            X_test_scaled,
            y_test,
            "Logistic Regression"
        ),
        get_model_metrics(
            rf_model,
            X_test,
            y_test,
            "Random Forest"
        )
    ]

    comparison_df = compare_models(results)

    print("\nModel comparison:")
    print(comparison_df)


if __name__ == "__main__":
    main()
