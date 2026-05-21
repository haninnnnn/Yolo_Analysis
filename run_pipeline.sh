#!/bin/bash

# Main execution script for YOLO Time Series Analysis Pipeline
# Run all components: data generation, analysis, and predictions

set -e

echo "=========================================="
echo "YOLO Time Series Analysis Pipeline"
echo "=========================================="

cd "$(dirname "$0")"

# Create data and results directories
mkdir -p data results

echo ""
echo "Step 1: Generating YOLO time series data..."
python scripts/generate_yolo_timeseries.py

echo ""
echo "Step 2: Performing statistical analysis..."
python scripts/analyze_timeseries.py

echo ""
echo "Step 3: Training predictive models..."
python scripts/predictive_model.py

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - data/yolo_timeseries.csv"
echo "  - data/yolo_timeseries.json"
echo "  - data/analysis_report.json"
echo "  - data/prediction_results.json"
echo "  - results/correlation_heatmap.png"
echo "  - results/trajectories.png"
echo "  - results/velocity_distribution.png"
echo "  - results/predictions_object_*.png"
echo ""
echo "Open results in: results/ directory"
