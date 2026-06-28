import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("dataset_2.csv")
df = df.dropna(subset=["Price"])

y = df["Price"]
feature_cols = ["Area_SqFt", "Rooms", "Build_Year", "Location", "Street_Type", "Furnishing", "Property_Type", "Has_Pool"]
X = df[feature_cols]

# Track which columns are categorical
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

for col in X.columns:
    if col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
    else:
        X[col] = X[col].fillna(X[col].median())

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
model_columns = list(X_encoded.columns)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

train_mae = mean_absolute_error(y_train, model.predict(X_train))
test_mae = mean_absolute_error(y_test, model.predict(X_test))

artifacts = {
    "model": model,
    "model_columns": model_columns,
    "categorical_cols": categorical_cols,  # Saved for API alignment
    "mae_metrics": {
        "training_mae": float(train_mae),
        "testing_mae": float(test_mae)
    }
}

with open("rf_model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)
