import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# model = joblib.load('xgboost_model.pkl') 
model = joblib.load('lgbm_model.pkl') 

test_data = pd.read_csv('../data/test_preprocessed.csv')
X_test = test_data.drop(columns=['active_power'])
y_test = test_data['active_power']

y_pred = model.predict(X_test)

def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    }

metrics = calculate_regression_metrics(y_test, y_pred)

print("Model evaluation results on test data:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

