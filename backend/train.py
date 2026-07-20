import os
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "home_model.joblib")
COLUMNS_PATH = os.path.join(SCRIPT_DIR, "model_columns.joblib")

DATA_FILE = os.path.join(SCRIPT_DIR, "dataset", "train.csv")
home_data = pd.read_csv(DATA_FILE)

features = [
    'Rooms', 'Type', 'Distance', 'Bathroom', 'Landsize', 'Regionname', 
]

X_raw = home_data[features]     # 6 columns
y = home_data['Price']
X = pd.get_dummies(X_raw)       # Many columns: 'Type' and 'Regionname' columns are one-hot encoded now.

home_model = HistGradientBoostingRegressor(random_state=1)

print ("Training...")
home_model.fit(X, y)

model_columns = list(X.columns)

joblib.dump(home_model, MODEL_PATH)
joblib.dump(model_columns, COLUMNS_PATH)

print(f"Files saved successfully in: {SCRIPT_DIR}")