"""
Build predictive models to forecast YOLO object movements.
Uses regression and time series forecasting techniques.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import json

class MovementPredictor:
    """Predict object movements based on historical trajectory"""
    
    def __init__(self, df, look_back=10, prediction_horizon=5):
        """
        Args:
            df: DataFrame with YOLO time series data
            look_back: Number of past frames to use for prediction
            prediction_horizon: Number of frames to predict ahead
        """
        self.df = df.sort_values(['object_id', 'frame']).copy()
        self.look_back = look_back
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        self.models = {}
        self.results = {}
        
    def create_sequences(self, object_id):
        """Create sequences for time series prediction"""
        obj_data = self.df[self.df['object_id'] == object_id].sort_values('frame').copy()
        
        # Features to predict: position and velocity
        features = ['center_x', 'center_y', 'vx', 'vy']
        feature_data = obj_data[features].dropna().values
        
        X = []
        y_x = []
        y_y = []
        
        for i in range(len(feature_data) - self.look_back - self.prediction_horizon):
            # Input: look_back frames
            X.append(feature_data[i:i + self.look_back].flatten())
            
            # Output: future position (prediction_horizon frames ahead)
            future_x = feature_data[i + self.look_back + self.prediction_horizon - 1, 0]  # center_x
            future_y = feature_data[i + self.look_back + self.prediction_horizon - 1, 1]  # center_y
            
            y_x.append(future_x)
            y_y.append(future_y)
        
        if len(X) == 0:
            return None, None, None, None
        
        return np.array(X), np.array(y_x), np.array(y_y), features
    
    def train_models(self):
        """Train prediction models for each object"""
        for obj_id in sorted(self.df['object_id'].unique()):
            print(f"\nTraining models for object {obj_id}...")
            
            X, y_x, y_y, features = self.create_sequences(obj_id)
            
            if X is None or len(X) < 20:
                print(f"  Insufficient data for object {obj_id}")
                continue
            
            # Split data
            X_train, X_test, y_x_train, y_x_test, y_y_train, y_y_test = train_test_split(
                X, y_x, y_y, test_size=0.2, random_state=42
            )
            
            # Scale input features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[obj_id] = scaler
            
            # Train multiple models
            self.models[obj_id] = {}
            
            # Model 1: Linear Regression
            models_to_train = {
                'Linear': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            }
            
            for model_name, model in models_to_train.items():
                # Train for X coordinate
                model_x = model.__class__(**model.get_params())
                model_x.fit(X_train_scaled, y_x_train)
                
                # Train for Y coordinate
                model_y = model.__class__(**model.get_params())
                model_y.fit(X_train_scaled, y_y_train)
                
                # Predictions
                y_x_pred = model_x.predict(X_test_scaled)
                y_y_pred = model_y.predict(X_test_scaled)
                
                # Metrics
                mse_x = mean_squared_error(y_x_test, y_x_pred)
                mse_y = mean_squared_error(y_y_test, y_y_pred)
                r2_x = r2_score(y_x_test, y_x_pred)
                r2_y = r2_score(y_y_test, y_y_pred)
                mae_x = mean_absolute_error(y_x_test, y_x_pred)
                mae_y = mean_absolute_error(y_y_test, y_y_pred)
                
                self.models[obj_id][model_name] = {
                    'model_x': model_x,
                    'model_y': model_y,
                    'metrics': {
                        'mse_x': float(mse_x),
                        'mse_y': float(mse_y),
                        'rmse_x': float(np.sqrt(mse_x)),
                        'rmse_y': float(np.sqrt(mse_y)),
                        'r2_x': float(r2_x),
                        'r2_y': float(r2_y),
                        'mae_x': float(mae_x),
                        'mae_y': float(mae_y),
                        'avg_r2': float((r2_x + r2_y) / 2)
                    }
                }
                
                print(f"  {model_name}: R² = {(r2_x + r2_y) / 2:.4f}, RMSE_X = {np.sqrt(mse_x):.4f}, RMSE_Y = {np.sqrt(mse_y):.4f}")
    
    def predict_next_positions(self, object_id, model_name='GradientBoosting', steps=5):
        """Predict next N positions for an object"""
        if object_id not in self.models or model_name not in self.models[object_id]:
            return None
        
        obj_data = self.df[self.df['object_id'] == object_id].sort_values('frame')
        features = ['center_x', 'center_y', 'vx', 'vy']
        latest_data = obj_data[features].dropna().values
        
        if len(latest_data) < self.look_back:
            return None
        
        scaler = self.scalers[object_id]
        model_dict = self.models[object_id][model_name]
        model_x = model_dict['model_x']
        model_y = model_dict['model_y']
        
        predictions = []
        current_sequence = latest_data[-self.look_back:].copy()
        
        for step in range(steps):
            # Normalize and predict
            X_input = current_sequence.flatten().reshape(1, -1)
            X_scaled = scaler.transform(X_input)
            
            pred_x = model_x.predict(X_scaled)[0]
            pred_y = model_y.predict(X_scaled)[0]
            
            predictions.append({
                'step': step + 1,
                'predicted_x': float(pred_x),
                'predicted_y': float(pred_y)
            })
            
            # Update sequence for next iteration
            # Estimate velocities from positions
            if len(predictions) == 1:
                dx = pred_x - current_sequence[-1, 0]
                dy = pred_y - current_sequence[-1, 1]
            else:
                dx = pred_x - predictions[-2]['predicted_x']
                dy = pred_y - predictions[-2]['predicted_y']
            
            # Create new data point and update sequence
            new_point = np.array([pred_x, pred_y, dx, dy])
            current_sequence = np.vstack([current_sequence[1:], new_point])
        
        return predictions
    
    def generate_formulas(self):
        """Generate simplified prediction formulas for each object"""
        formulas = {}
        
        for obj_id in sorted(self.models.keys()):
            obj_data = self.df[self.df['object_id'] == obj_id].sort_values('frame')
            
            # Simple linear regression formula
            X = obj_data[['vx', 'vy']].dropna().values
            y_x = obj_data[['center_x']].dropna().values
            y_y = obj_data[['center_y']].dropna().values
            
            if len(X) < 10:
                continue
            
            # Fit simple model
            model_x = LinearRegression()
            model_x.fit(X, y_x)
            
            model_y = LinearRegression()
            model_y.fit(X, y_y)
            
            # Get latest values
            latest = obj_data.iloc[-1]
            
            formulas[f'object_{obj_id}'] = {
                'type': 'Linear Velocity Model',
                'x_formula': f"x_next ≈ {latest['center_x']:.4f} + {model_x.coef_[0][0]:.4f}*vx + {model_x.coef_[0][1]:.4f}*vy + {model_x.intercept_[0]:.4f}",
                'y_formula': f"y_next ≈ {latest['center_y']:.4f} + {model_y.coef_[0][0]:.4f}*vx + {model_y.coef_[0][1]:.4f}*vy + {model_y.intercept_[0]:.4f}",
                'coefficients': {
                    'x_coef_vx': float(model_x.coef_[0][0]),
                    'x_coef_vy': float(model_x.coef_[0][1]),
                    'x_intercept': float(model_x.intercept_[0]),
                    'y_coef_vx': float(model_y.coef_[0][0]),
                    'y_coef_vy': float(model_y.coef_[0][1]),
                    'y_intercept': float(model_y.intercept_[0])
                }
            }
        
        return formulas
    
    def save_results(self, output_file='../data/prediction_results.json'):
        """Save all models and results"""
        results = {
            'look_back': self.look_back,
            'prediction_horizon': self.prediction_horizon,
            'object_models': {}
        }
        
        for obj_id in self.models:
            results['object_models'][f'object_{obj_id}'] = {}
            for model_name, model_dict in self.models[obj_id].items():
                results['object_models'][f'object_{obj_id}'][model_name] = model_dict['metrics']
        
        # Add predictions for next 10 frames
        for obj_id in self.models:
            preds = self.predict_next_positions(obj_id, 'GradientBoosting', steps=10)
            if preds:
                results['object_models'][f'object_{obj_id}']['next_positions'] = preds
        
        # Add formulas
        results['formulas'] = self.generate_formulas()
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")
        return results


def visualize_predictions(predictor, output_dir='../results'):
    """Visualize predictions vs actual data"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for obj_id in list(predictor.models.keys())[:3]:  # First 3 objects
        obj_data = predictor.df[predictor.df['object_id'] == obj_id].sort_values('frame')
        
        predictions = predictor.predict_next_positions(obj_id, 'GradientBoosting', steps=20)
        
        if predictions is None:
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Historical and predicted trajectory
        ax1.plot(obj_data['center_x'], obj_data['center_y'], 'b-', label='Historical', linewidth=2, marker='o', markersize=3)
        
        pred_x = [p['predicted_x'] for p in predictions]
        pred_y = [p['predicted_y'] for p in predictions]
        ax1.plot(pred_x, pred_y, 'r--', label='Predicted', linewidth=2, marker='s', markersize=3)
        
        ax1.scatter([obj_data['center_x'].iloc[-1]], [obj_data['center_y'].iloc[-1]], c='green', s=100, marker='*', label='Current', zorder=5)
        
        ax1.set_xlabel('Center X')
        ax1.set_ylabel('Center Y')
        ax1.set_title(f'Object {obj_id} - Trajectory Prediction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Position over frame
        ax2.plot(range(len(obj_data)), obj_data['center_x'], 'b-', label='Historical X', marker='o', markersize=3)
        last_frame = obj_data['frame'].iloc[-1]
        pred_frames = range(int(last_frame) + 1, int(last_frame) + len(predictions) + 1)
        ax2.plot(pred_frames, pred_x, 'r--', label='Predicted X', marker='s', markersize=3)
        
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Center X Position')
        ax2.set_title(f'Object {obj_id} - Position Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/predictions_object_{obj_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/predictions_object_{obj_id}.png")


if __name__ == "__main__":
    # Load data
    print("Loading YOLO time series data...")
    df = pd.read_csv('../data/yolo_timeseries.csv')
    
    # Initialize predictor
    predictor = MovementPredictor(df, look_back=10, prediction_horizon=5)
    
    # Train models
    print("\n" + "="*60)
    print("TRAINING PREDICTION MODELS")
    print("="*60)
    predictor.train_models()
    
    # Generate and save results
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS AND FORMULAS")
    print("="*60)
    results = predictor.save_results()
    
    # Print sample formulas
    print("\nSample Prediction Formulas:")
    for obj, formula in list(results['formulas'].items())[:2]:
        print(f"\n{obj}:")
        print(f"  {formula['x_formula']}")
        print(f"  {formula['y_formula']}")
    
    # Create visualizations
    print("\nGenerating prediction visualizations...")
    visualize_predictions(predictor)
    
    print("\n✓ Prediction models complete!")
