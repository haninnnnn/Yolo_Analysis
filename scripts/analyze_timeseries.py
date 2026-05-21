"""
Analyze YOLO coordinate time series data.
Compute correlation, covariance, and statistical features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import json

class YOLOTimeSeriesAnalyzer:
    """Analyze YOLO time series data"""
    
    def __init__(self, df):
        """
        Args:
            df: DataFrame with YOLO time series data
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        
    def compute_correlation_matrix(self, object_id=None):
        """Compute correlation matrix for position features"""
        if object_id is not None:
            data = self.df[self.df['object_id'] == object_id]
        else:
            data = self.df
        
        features = ['center_x', 'center_y', 'width', 'height', 'confidence']
        
        # Remove NaN values
        feature_data = data[features].dropna()
        
        corr_matrix = feature_data.corr()
        return corr_matrix
    
    def compute_covariance_matrix(self, object_id=None):
        """Compute covariance matrix"""
        if object_id is not None:
            data = self.df[self.df['object_id'] == object_id]
        else:
            data = self.df
        
        features = ['center_x', 'center_y', 'width', 'height', 'confidence']
        feature_data = data[features].dropna()
        
        cov_matrix = feature_data.cov()
        return cov_matrix
    
    def compute_autocorrelation(self, object_id, lags=50):
        """Compute autocorrelation for position time series"""
        obj_data = self.df[self.df['object_id'] == object_id].sort_values('frame')
        
        results = {}
        for feature in ['center_x', 'center_y', 'speed']:
            if feature in obj_data.columns:
                series = obj_data[feature].dropna().values
                acf_values = [1.0] + [
                    np.corrcoef(series[:-lag], series[lag:])[0, 1] 
                    for lag in range(1, min(lags + 1, len(series)))
                ]
                results[feature] = acf_values
        
        return results
    
    def compute_statistical_summary(self, object_id=None):
        """Compute statistical summary for each feature"""
        if object_id is not None:
            data = self.df[self.df['object_id'] == object_id]
        else:
            data = self.df
        
        features = ['center_x', 'center_y', 'width', 'height', 'speed', 'confidence']
        
        summary = {}
        for feature in features:
            if feature in data.columns:
                series = data[feature].dropna()
                summary[feature] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'median': series.median(),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75),
                    'skewness': stats.skew(series),
                    'kurtosis': stats.kurtosis(series)
                }
        
        return summary
    
    def detect_movement_patterns(self, object_id, window_size=30):
        """Detect movement patterns (acceleration, deceleration, direction changes)"""
        obj_data = self.df[self.df['object_id'] == object_id].sort_values('frame')
        
        if 'vx' not in obj_data.columns or 'vy' not in obj_data.columns:
            return None
        
        vx = obj_data['vx'].dropna().values
        vy = obj_data['vy'].dropna().values
        
        # Smooth velocity with moving average
        vx_smooth = pd.Series(vx).rolling(window=window_size).mean().values
        vy_smooth = pd.Series(vy).rolling(window=window_size).mean().values
        
        # Detect turning points (direction changes)
        angle = np.arctan2(vy_smooth, vx_smooth)
        angle_diff = np.diff(angle)
        
        # Count direction changes (significant angle changes)
        turning_points = np.where(np.abs(angle_diff) > 0.3)[0]
        
        return {
            'mean_vx': np.nanmean(vx),
            'mean_vy': np.nanmean(vy),
            'std_vx': np.nanstd(vx),
            'std_vy': np.nanstd(vy),
            'turning_points': len(turning_points),
            'avg_speed': np.nanmean(np.sqrt(vx**2 + vy**2))
        }
    
    def generate_analysis_report(self, output_file=None):
        """Generate comprehensive analysis report"""
        report = {
            'num_frames': int(self.df['frame'].max() + 1),
            'num_objects': int(self.df['object_id'].nunique()),
            'total_detections': len(self.df),
            'features': ['center_x', 'center_y', 'width', 'height', 'speed'],
            'global_correlation': self.compute_correlation_matrix().to_dict(),
            'object_analysis': {}
        }
        
        for obj_id in sorted(self.df['object_id'].unique()):
            report['object_analysis'][f'object_{int(obj_id)}'] = {
                'correlation': self.compute_correlation_matrix(obj_id).to_dict(),
                'statistics': self.compute_statistical_summary(obj_id),
                'movement_patterns': self.detect_movement_patterns(obj_id),
            }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to {output_file}")
        
        return report


def visualize_analysis(df, output_dir='../results'):
    """Create visualization plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    features = ['center_x', 'center_y', 'width', 'height', 'confidence']
    corr_matrix = df[features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    plt.title('YOLO Features Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/correlation_heatmap.png")
    
    # 2. Position trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for obj_id in df['object_id'].unique():
        obj_data = df[df['object_id'] == obj_id].sort_values('frame')
        ax1.plot(obj_data['center_x'], obj_data['center_y'], marker='o', markersize=2, alpha=0.6, label=f'Object {obj_id}')
    
    ax1.set_xlabel('Center X')
    ax1.set_ylabel('Center Y')
    ax1.set_title('Object Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3. Position over time
    for obj_id in df['object_id'].unique()[:3]:  # First 3 objects
        obj_data = df[df['object_id'] == obj_id].sort_values('frame')
        ax2.plot(obj_data['frame'], obj_data['center_x'], label=f'Object {obj_id}', alpha=0.7)
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Center X Position')
    ax2.set_title('Position Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/trajectories.png")
    
    # 3. Velocity distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'vx' in df.columns and 'vy' in df.columns:
        df['vx'].hist(bins=50, ax=ax1, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Velocity X')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Velocity X Distribution')
        ax1.grid(True, alpha=0.3)
        
        df['vy'].hist(bins=50, ax=ax2, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Velocity Y')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Velocity Y Distribution')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/velocity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/velocity_distribution.png")


if __name__ == "__main__":
    # Load data
    print("Loading YOLO time series data...")
    df = pd.read_csv('../data/yolo_timeseries.csv')
    
    # Initialize analyzer
    analyzer = YOLOTimeSeriesAnalyzer(df)
    
    # Generate analysis report
    print("\nGenerating analysis report...")
    report = analyzer.generate_analysis_report('../data/analysis_report.json')
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total frames: {report['num_frames']}")
    print(f"Total objects: {report['num_objects']}")
    print(f"Total detections: {report['total_detections']}")
    
    print("\nGlobal Correlation:")
    global_corr = pd.DataFrame(report['global_correlation'])
    print(global_corr)
    
    # Statistics for first object
    if 'object_0' in report['object_analysis']:
        print("\nObject 0 Statistics:")
        stats = report['object_analysis']['object_0']['statistics']
        for feature, stat in stats.items():
            print(f"\n{feature}:")
            for key, val in stat.items():
                print(f"  {key}: {val:.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    import os
    os.makedirs('../results', exist_ok=True)
    visualize_analysis(df)
    
    print("\n✓ Analysis complete!")
