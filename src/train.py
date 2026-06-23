import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess_data
from evaluate import evaluate_model

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

    print("Logistic Regression")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_scaled, y_train)

    evaluate_model(log_model, X_test_scaled, y_test)

    print("\nRandom Forest")
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    evaluate_model(rf_model, X_test, y_test)


if __name__ == "__main__":
    main()
