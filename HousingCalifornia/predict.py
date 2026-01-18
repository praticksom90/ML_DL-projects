import pandas as pd
import joblib

# Load model & pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Load new data
input_data = pd.read_csv("input.csv")

# Transform + predict
prepared_data = pipeline.transform(input_data)
predictions = model.predict(prepared_data)

input_data["predicted_house_value"] = predictions
input_data.to_csv("output.csv", index=False)

print("Predictions saved to output.csv")
