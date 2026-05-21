# YOLO v7 Coordinate Time Series Analysis

This project generates synthetic YOLO v7 detection time series data and performs comprehensive statistical analysis and predictive modeling on object movements.

## Project Structure

```
yolo-analysis/
├── data/                          # Generated data and analysis results
│   ├── yolo_timeseries.csv       # Main time series dataset
│   ├── yolo_timeseries.json      # Frame-based detection format
│   ├── analysis_report.json      # Statistical analysis report
│   └── prediction_results.json   # Model predictions and formulas
├── scripts/                       # Python scripts
│   ├── generate_yolo_timeseries.py    # Generate synthetic data
│   ├── analyze_timeseries.py          # Statistical analysis
│   └── predictive_model.py            # ML prediction models
├── notebooks/                     # Jupyter notebooks for exploration
├── results/                       # Generated visualizations
└── requirements.txt              # Python dependencies
```

## Features

### 1. **Data Generation** (`generate_yolo_timeseries.py`)
- Generates realistic YOLO coordinate data with:
  - Object trajectories with realistic physics (bouncing, friction)
  - Position coordinates (center_x, center_y)
  - Box dimensions (width, height)
  - Object classes (COCO 0-79)
  - Confidence scores
  - Derived features: velocity (vx, vy) and acceleration (ax, ay)

### 2. **Statistical Analysis** (`analyze_timeseries.py`)
- **Correlation Analysis**: Feature correlation matrices
- **Covariance Analysis**: Covariance between features
- **Autocorrelation**: Time series autocorrelation for each object
- **Statistical Summaries**: Mean, std, skewness, kurtosis, quartiles
- **Movement Pattern Detection**: Identifies turning points and direction changes
- **Visualizations**: Heatmaps, trajectories, distributions

### 3. **Predictive Modeling** (`predictive_model.py`)
- Multiple ML models:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
- Predicts future positions (center_x, center_y) based on historical trajectory
- Generates prediction formulas for each object
- Evaluates models with: R², RMSE, MAE
- Visualizes predictions vs actual trajectories

## Installation

```bash
cd /Users/user/Documents/yolo-analysis
pip install -r requirements.txt
```

## Usage

### 1. Generate Data
```bash
cd scripts
python generate_yolo_timeseries.py
```

This creates:
- `data/yolo_timeseries.csv`: Time series data (500 frames × 5 objects)
- `data/yolo_timeseries.json`: Frame-based JSON format

### 2. Run Statistical Analysis
```bash
python analyze_timeseries.py
```

This generates:
- `data/analysis_report.json`: Detailed analysis report
- `results/correlation_heatmap.png`: Correlation matrix visualization
- `results/trajectories.png`: Object trajectories plot
- `results/velocity_distribution.png`: Velocity distribution plots

### 3. Train Predictive Models
```bash
python predictive_model.py
```

This produces:
- `data/prediction_results.json`: Model metrics and formulas
- `results/predictions_object_*.png`: Prediction visualizations

## Data Format

### CSV Format
```
frame,timestamp,object_id,time_seconds,center_x,center_y,width,height,class,confidence,vx,vy,speed,ax,ay
0,2026-05-12 ...,0,0.0,0.5234,0.4876,0.1234,0.1567,45,0.9823,0.0012,-0.0034,0.0037,0.0001,-0.0002
...
```

### JSON Format (Frame-based)
```json
[
  {
    "frame": 0,
    "timestamp": "2026-05-12T...",
    "detections": [
      {
        "object_id": 0,
        "center_x": 0.5234,
        "center_y": 0.4876,
        "width": 0.1234,
        "height": 0.1567,
        "class": 45,
        "confidence": 0.9823
      }
    ]
  }
]
```

## Analysis Outputs

### Correlation Matrix
Shows how different features correlate:
- High correlation between speed and velocity components
- Position stability across sequences

### Statistical Summaries
For each feature:
- Mean, Std Dev, Min, Max
- Quartiles (Q25, Q75)
- Skewness and Kurtosis

### Movement Patterns
- Mean/Std of velocity components
- Turning point detection
- Average speed metrics

## Prediction Formulas

The models generate linear prediction formulas for quick inference:

```
Object 0:
x_next ≈ 0.5234 + 0.8234*vx + 0.1123*vy - 0.0034
y_next ≈ 0.4876 + 0.1567*vx + 0.8901*vy - 0.0056
```

These coefficients can be embedded directly into applications for fast predictions.

## Model Performance

Expected metrics for synthetic data:
- **R² Score**: 0.85-0.95 (high linear predictability)
- **RMSE**: 0.02-0.05 (position accuracy)
- **MAE**: 0.015-0.03 (average error)

Better performance indicates more predictable movement patterns.

## Customization

### Parameters in `generate_yolo_timeseries.py`
```python
generator = YOLOTimeSeriesGenerator(
    num_objects=5,      # Number of tracked objects
    num_frames=500,     # Total frames to generate
    frame_rate=30       # Frames per second
)
```

### Parameters in `predictive_model.py`
```python
predictor = MovementPredictor(
    df,
    look_back=10,              # Past frames to use
    prediction_horizon=5       # Frames to predict ahead
)
```

## Next Steps

1. **Replace synthetic data** with real YOLO detections from your camera/videos
2. **Tune model parameters** based on your specific movement patterns
3. **Implement real-time prediction** using the generated formulas
4. **Add Kalman filtering** for smoother predictions
5. **Experiment with LSTM models** for sequence-to-sequence prediction

## Important Notes

- The synthetic data includes realistic physics and randomness
- All coordinates are normalized (0-1 range)
- Velocity is in normalized units per second
- Models are trained on 80% of data, validated on 20%
- Consider temporal autocorrelation when interpreting results

## References

- YOLO v7 format: https://github.com/WongKinYiu/yolov7
- COCO dataset classes: https://cocodataset.org/
- Time series forecasting: scikit-learn, pandas documentation
