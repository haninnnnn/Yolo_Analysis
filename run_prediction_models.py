#!/usr/bin/env python
"""
Train prediction models for YOLO time series object movements.
"""

import pandas as pd
import numpy as np
import json
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("=" * 70)
print("TRAINING PREDICTION MODELS")
print("=" * 70)

# Load data
print("\nLoading data...")
df = pd.read_csv('data/yolo_timeseries.csv')

# Parameters  
look_back = 10
prediction_horizon = 5

def create_sequences(data, look_back=10, prediction_horizon=5):
    """Create sequences for time series prediction"""
    X, y_x, y_y = [], [], []
    
    for i in range(len(data) - look_back - prediction_horizon):
        X.append(data[i:i + look_back].flatten())
        future_idx = i + look_back + prediction_horizon - 1
        y_x.append(data[future_idx, 0])  # center_x
        y_y.append(data[future_idx, 1])  # center_y
    
    return np.array(X), np.array(y_x), np.array(y_y)

# Train models
models_by_object = {}
metrics_by_object = {}
prediction_formulas = {}

print("\nTraining models for each object...")
print("-" * 70)

for obj_id in sorted(df['object_id'].unique()):
    obj_data = df[df['object_id'] == obj_id].sort_values('frame')
    
    feature_cols = ['center_x', 'center_y', 'vx', 'vy']
    feature_data = obj_data[feature_cols].dropna().values
    
    if len(feature_data) < look_back + prediction_horizon + 10:
        print(f"  Object {int(obj_id)}: Insufficient data")
        continue
    
    # Create sequences
    X, y_x, y_y = create_sequences(feature_data, look_back, prediction_horizon)
    
    # Split data
    X_train, X_test, y_x_train, y_x_test, y_y_train, y_y_test = train_test_split(
        X, y_x, y_y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=30, max_depth=4, random_state=42)
    }
    
    object_models = {}
    best_r2 = -1
    best_model_name = None
    
    for model_name, base_model in models.items():
        # Train for X
        model_x = base_model.__class__(**base_model.get_params())
        model_x.fit(X_train_scaled, y_x_train)
        
        # Train for Y
        model_y = base_model.__class__(**base_model.get_params())
        model_y.fit(X_train_scaled, y_y_train)
        
        # Predict
        y_x_pred = model_x.predict(X_test_scaled)
        y_y_pred = model_y.predict(X_test_scaled)
        
        # Metrics
        r2_x = r2_score(y_x_test, y_x_pred)
        r2_y = r2_score(y_y_test, y_y_pred)
        rmse_x = np.sqrt(mean_squared_error(y_x_test, y_x_pred))
        rmse_y = np.sqrt(mean_squared_error(y_y_test, y_y_pred))
        mae_x = mean_absolute_error(y_x_test, y_x_pred)
        mae_y = mean_absolute_error(y_y_test, y_y_pred)
        
        avg_r2 = (r2_x + r2_y) / 2
        
        object_models[model_name] = {
            'model_x': model_x,
            'model_y': model_y,
            'scaler': scaler,
            'r2_x': float(r2_x),
            'r2_y': float(r2_y),
            'rmse_x': float(rmse_x),
            'rmse_y': float(rmse_y),
            'mae_x': float(mae_x),
            'mae_y': float(mae_y),
        }
        
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model_name = model_name
        
        print(f"  Object {int(obj_id)} - {model_name:15s}: R²={avg_r2:.4f}, RMSE_X={rmse_x:.4f}, RMSE_Y={rmse_y:.4f}")
    
    models_by_object[obj_id] = object_models
    metrics_by_object[obj_id] = {
        'best_model': best_model_name,
        'best_r2': float(best_r2),
        'look_back': look_back,
        'prediction_horizon': prediction_horizon
    }
    
    # Generate linear formula
    X_vel = obj_data[['vx', 'vy']].dropna().values
    y_x_all = obj_data[['center_x']].dropna().values
    y_y_all = obj_data[['center_y']].dropna().values
    
    if len(X_vel) > 5:
        lr_x = LinearRegression()
        lr_x.fit(X_vel, y_x_all)
        
        lr_y = LinearRegression()
        lr_y.fit(X_vel, y_y_all)
        
        latest = obj_data.iloc[-1]
        
        prediction_formulas[f'object_{int(obj_id)}'] = {
            'formula_x': f"x_next = {latest['center_x']:.4f} + {lr_x.coef_[0][0]:.4f}*vx + {lr_x.coef_[0][1]:.4f}*vy + {lr_x.intercept_[0]:.4f}",
            'formula_y': f"y_next = {latest['center_y']:.4f} + {lr_y.coef_[0][0]:.4f}*vx + {lr_y.coef_[0][1]:.4f}*vy + {lr_y.intercept_[0]:.4f}",
            'x_coef_vx': float(lr_x.coef_[0][0]),
            'x_coef_vy': float(lr_x.coef_[0][1]),
            'x_intercept': float(lr_x.intercept_[0]),
            'y_coef_vx': float(lr_y.coef_[0][0]),
            'y_coef_vy': float(lr_y.coef_[0][1]),
            'y_intercept': float(lr_y.intercept_[0])
        }

# Save results
print("\nSaving results...")

results = {
    'look_back': look_back,
    'prediction_horizon': prediction_horizon,
    'object_models': {},
    'formulas': prediction_formulas
}

for obj_id in models_by_object:
    results['object_models'][f'object_{int(obj_id)}'] = {}
    for model_name, model_dict in models_by_object[obj_id].items():
        results['object_models'][f'object_{int(obj_id)}'][model_name] = {
            'r2_x': model_dict['r2_x'],
            'r2_y': model_dict['r2_y'],
            'rmse_x': model_dict['rmse_x'],
            'rmse_y': model_dict['rmse_y'],
            'mae_x': model_dict['mae_x'],
            'mae_y': model_dict['mae_y'],
        }

with open('data/prediction_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("   ✓ Saved: data/prediction_results.json")

# Print summary
print("\n" + "=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)

print(f"\n{'Object':<10} {'Best Model':<18} {'R² Score':<12} {'RMSE_X':<12} {'RMSE_Y':<12}")
print("-" * 70)

for obj_id in sorted(models_by_object.keys()):
    best_model = metrics_by_object[obj_id]['best_model']
    best_r2 = metrics_by_object[obj_id]['best_r2']
    
    model_dict = models_by_object[obj_id][best_model]
    rmse_x = model_dict['rmse_x']
    rmse_y = model_dict['rmse_y']
    
    print(f"{int(obj_id):<10} {best_model:<18} {best_r2:<12.4f} {rmse_x:<12.4f} {rmse_y:<12.4f}")

print("\n" + "=" * 70)
print("PREDICTION FORMULAS (Linear Velocity Model)")
print("=" * 70)

for obj_id, formula_dict in prediction_formulas.items():
    print(f"\n{obj_id}:")
    print(f"  {formula_dict['formula_x']}")
    print(f"  {formula_dict['formula_y']}")

print("\n" + "=" * 70)
print("✓ MODEL TRAINING COMPLETE!")
print("=" * 70)
