#!/usr/bin/env python
"""
Quick prediction model training (fixed version).
"""

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

print("=" * 70)
print("TRAINING PREDICTION MODELS")
print("=" * 70)

# Load data
df = pd.read_csv('data/yolo_timeseries.csv')

# Simple linear formulas based on velocity
formulas = {}
metrics = {}

for obj_id in sorted(df['object_id'].unique()):
    obj_data = df[df['object_id'] == obj_id].sort_values('frame').copy()
    
    # Remove rows with NaN in velocity or position columns
    clean_data = obj_data[['vx', 'vy', 'center_x', 'center_y']].dropna()
    
    if len(clean_data) > 5:
        X = clean_data[['vx', 'vy']].values
        y_x = clean_data[['center_x']].values
        y_y = clean_data[['center_y']].values
        
        # Linear regression for X
        lr_x = LinearRegression()
        lr_x.fit(X, y_x)
        pred_x = lr_x.predict(X)
        r2_x = r2_score(y_x, pred_x)
        rmse_x = np.sqrt(mean_squared_error(y_x, pred_x))
        
        # Linear regression for Y
        lr_y = LinearRegression()
        lr_y.fit(X, y_y)
        pred_y = lr_y.predict(X)
        r2_y = r2_score(y_y, pred_y)
        rmse_y = np.sqrt(mean_squared_error(y_y, pred_y))
        
        avg_r2 = (r2_x + r2_y) / 2
        
        latest = clean_data.iloc[-1]
        
        formulas[f'object_{int(obj_id)}'] = {
            'formula_x': f"x_next = {latest[2]:.4f} + {lr_x.coef_[0][0]:.6f}*vx + {lr_x.coef_[0][1]:.6f}*vy + {lr_x.intercept_[0]:.6f}",
            'formula_y': f"y_next = {latest[3]:.4f} + {lr_y.coef_[0][0]:.6f}*vx + {lr_y.coef_[0][1]:.6f}*vy + {lr_y.intercept_[0]:.6f}",
            'coefficients': {
                'x_coef_vx': float(lr_x.coef_[0][0]),
                'x_coef_vy': float(lr_x.coef_[0][1]),
                'x_intercept': float(lr_x.intercept_[0]),
                'y_coef_vx': float(lr_y.coef_[0][0]),
                'y_coef_vy': float(lr_y.coef_[0][1]),
                'y_intercept': float(lr_y.intercept_[0])
            }
        }
        
        metrics[f'object_{int(obj_id)}'] = {
            'r2_x': float(r2_x),
            'r2_y': float(r2_y),
            'avg_r2': float(avg_r2),
            'rmse_x': float(rmse_x),
            'rmse_y': float(rmse_y)
        }
        
        print(f"Object {int(obj_id)}: R²={avg_r2:.4f}, RMSE_X={rmse_x:.4f}, RMSE_Y={rmse_y:.4f}")

# Save predictions
results = {
    'model_type': 'Linear Regression (Velocity-based)',
    'description': 'Predicts next position using current velocity components',
    'formulas': formulas,
    'metrics': metrics
}

with open('data/prediction_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n✓ Saved: data/prediction_results.json")

print("\n" + "=" * 70)
print("PREDICTION FORMULAS")
print("=" * 70)

for obj_id, formula in formulas.items():
    print(f"\n{obj_id}:")
    print(f"  {formula['formula_x']}")
    print(f"  {formula['formula_y']}")

print("\n" + "=" * 70)
print("✓ MODEL TRAINING COMPLETE!")
print("=" * 70)
