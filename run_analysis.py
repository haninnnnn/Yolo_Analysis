#!/usr/bin/env python
"""
Quick analysis runner for YOLO time series data.
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("YOLO TIME SERIES DATA - COMPLETE ANALYSIS")
print("=" * 70)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('data/yolo_timeseries.csv')

print(f"   Data Shape: {df.shape}")
print(f"   Total Detections: {len(df)}")
print(f"   Total Frames: {int(df['frame'].max() + 1)}")
print(f"   Objects Tracked: {int(df['object_id'].nunique())}")
print(f"   Time Range: {df['time_seconds'].min():.2f} - {df['time_seconds'].max():.2f} seconds")

# Feature statistics
print("\n2. FEATURE STATISTICS")
print("-" * 70)
features = ['center_x', 'center_y', 'width', 'height', 'confidence', 'speed']
stats_df = df[features].describe().round(4)
print(stats_df)

# Correlation analysis
print("\n3. CORRELATION MATRIX (Key Features)")
print("-" * 70)
corr_features = ['center_x', 'center_y', 'width', 'height', 'confidence', 'vx', 'vy', 'speed']
correlation = df[corr_features].corr().round(4)
print(correlation)

# Covariance analysis
print("\n4. COVARIANCE MATRIX (Key Features)")
print("-" * 70)
covariance = df[corr_features].cov().round(6)
print(covariance)

# Movement statistics
print("\n5. MOVEMENT STATISTICS BY OBJECT")
print("-" * 70)
for obj_id in sorted(df['object_id'].unique()):
    obj_data = df[df['object_id'] == obj_id]
    print(f"\nObject {int(obj_id)}:")
    print(f"  Frames tracked: {obj_data['frame'].nunique()}")
    print(f"  Mean Speed: {obj_data['speed'].mean():.4f}")
    print(f"  Max Speed: {obj_data['speed'].max():.4f}")
    print(f"  Std Speed: {obj_data['speed'].std():.4f}")
    print(f"  Mean Vx: {obj_data['vx'].mean():.4f}")
    print(f"  Mean Vy: {obj_data['vy'].mean():.4f}")
    print(f"  Avg Confidence: {obj_data['confidence'].mean():.4f}")

# Create visualizations
print("\n6. Generating visualizations...")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'}, square=True)
plt.title('YOLO Features - Correlation Matrix')
plt.tight_layout()
plt.savefig('results/01_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/01_correlation_heatmap.png")

# Trajectories
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for obj_id in df['object_id'].unique():
    obj_data = df[df['object_id'] == obj_id].sort_values('frame')
    axes[0].plot(obj_data['center_x'], obj_data['center_y'], marker='o', markersize=2, alpha=0.6, label=f'Object {int(obj_id)}')

axes[0].set_xlabel('Center X')
axes[0].set_ylabel('Center Y')
axes[0].set_title('Object Trajectories')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Position over time
for obj_id in df['object_id'].unique()[:3]:
    obj_data = df[df['object_id'] == obj_id].sort_values('frame')
    axes[1].plot(obj_data['frame'], obj_data['center_x'], label=f'Object {int(obj_id)}', alpha=0.7)

axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Center X Position')
axes[1].set_title('X Position Over Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/02_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/02_trajectories.png")

# Velocity distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['vx'].hist(bins=50, ax=axes[0], alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Velocity X (vx)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Velocity X Distribution')
axes[0].grid(True, alpha=0.3)

df['vy'].hist(bins=50, ax=axes[1], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Velocity Y (vy)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Velocity Y Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/03_velocity_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/03_velocity_distribution.png")

# Speed comparison
fig, ax = plt.subplots(figsize=(10, 6))
for obj_id in df['object_id'].unique():
    obj_data = df[df['object_id'] == obj_id]
    ax.hist(obj_data['speed'].dropna(), bins=30, alpha=0.5, label=f'Object {int(obj_id)}')
ax.set_xlabel('Speed')
ax.set_ylabel('Frequency')
ax.set_title('Speed Distribution by Object')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/04_speed_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/04_speed_distribution.png")

# Save detailed report
print("\n7. Saving analysis report...")
report = {
    'summary': {
        'total_detections': len(df),
        'total_frames': int(df['frame'].max() + 1),
        'objects_tracked': int(df['object_id'].nunique()),
        'time_range': f"{df['time_seconds'].min():.2f} - {df['time_seconds'].max():.2f} seconds"
    },
    'feature_statistics': stats_df.to_dict(),
    'correlation_matrix': correlation.to_dict(),
    'covariance_matrix': covariance.to_dict(),
    'movement_statistics': {}
}

# Add per-object statistics
for obj_id in sorted(df['object_id'].unique()):
    obj_data = df[df['object_id'] == obj_id]
    report['movement_statistics'][f'object_{int(obj_id)}'] = {
        'frames_tracked': int(obj_data['frame'].nunique()),
        'mean_speed': float(obj_data['speed'].mean()),
        'max_speed': float(obj_data['speed'].max()),
        'std_speed': float(obj_data['speed'].std()),
        'mean_vx': float(obj_data['vx'].mean()),
        'mean_vy': float(obj_data['vy'].mean()),
        'avg_confidence': float(obj_data['confidence'].mean())
    }

with open('data/analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print("   ✓ Saved: data/analysis_report.json")

print("\n" + "=" * 70)
print("✓ ANALYSIS COMPLETE!")
print("=" * 70)
print("\nGenerated files:")
print("  • data/yolo_timeseries.csv (2.5MB)")
print("  • data/yolo_timeseries.json (683KB)")
print("  • data/analysis_report.json")
print("  • results/01_correlation_heatmap.png")
print("  • results/02_trajectories.png")
print("  • results/03_velocity_distribution.png")
print("  • results/04_speed_distribution.png")
print("\nNext step: Train prediction models using predictive_model.py")
