import pandas as pd

def clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def clean_total_charges(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df

def encode_Yes_No_features(df, feature):
    df = df.copy()
    df[feature] = df[feature].map({
        "No": 0,
        "Yes": 1
    })
    return df


def prepare_target(df):
    return encode_Yes_No_features(df, "Churn")

def drop_unused_columns(df):
    df = df.copy()
    return df.drop(columns=["customerID"])

def encode_binary_features(df):
    df = df.copy()
    yes_no_features = [
        "Partner",
        "Dependents",
        "PaperlessBilling",
        "PhoneService"
    ]
    for feature in yes_no_features:
        df = encode_Yes_No_features(df, feature)
    df["gender"] = df["gender"].map({
        "Female": 0,
        "Male": 1
    })
    return df


def encode_categorical_features(df):
    df = df.copy()
    categorical_features = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]
    df = pd.get_dummies(
        df,
        columns=categorical_features,
        drop_first=True
    )
    return df

def preprocess_data(df):
    df = clean_column_names(df)
    df = clean_total_charges(df)
    df = prepare_target(df)
    df = drop_unused_columns(df)
    df = encode_binary_features(df)
    df = encode_categorical_features(df)
    return df



