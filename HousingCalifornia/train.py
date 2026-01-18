import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# -----------------------------
# 1. Load dataset
# -----------------------------
housing = pd.read_csv("housing.csv")


# -----------------------------
# 2. Create income category for stratified split
# -----------------------------
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    train_set = housing.loc[train_idx].drop("income_cat", axis=1)
    test_set = housing.loc[test_idx].drop("income_cat", axis=1)


# -----------------------------
# 3. Separate features and labels
# -----------------------------
X_train = train_set.drop("median_house_value", axis=1)
y_train = train_set["median_house_value"]

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"]


# -----------------------------
# 4. Preprocessing pipeline
# -----------------------------
num_features = X_train.drop("ocean_proximity", axis=1).columns
cat_features = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])


X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)


# -----------------------------
# 5. Train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_prepared, y_train)


# -----------------------------
# 6. Evaluate model
# -----------------------------
predictions = model.predict(X_test_prepared)

mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5

print("Test RMSE:", rmse)


# -----------------------------
# 7. Save model and pipeline
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(full_pipeline, "pipeline.pkl")

print("Model and pipeline saved successfully.")
