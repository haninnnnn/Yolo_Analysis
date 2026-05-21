"""
Generate synthetic YOLO v7 coordinate time series data.
YOLO coordinates format: (center_x, center_y, width, height, class, confidence)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class YOLOTimeSeriesGenerator:
    """Generate realistic YOLO detection time series data"""
    
    def __init__(self, num_objects=5, num_frames=500, frame_rate=30):
        """
        Args:
            num_objects: Number of objects to track
            num_frames: Number of frames to generate
            frame_rate: Frames per second
        """
        self.num_objects = num_objects
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.frame_duration = 1.0 / frame_rate
        
        # Initialize object trajectories with random starting positions
        self.objects = self._init_objects()
        
    def _init_objects(self):
        """Initialize object trajectories with random velocities"""
        objects = {}
        for obj_id in range(self.num_objects):
            objects[obj_id] = {
                'x': np.random.uniform(0.1, 0.9),
                'y': np.random.uniform(0.1, 0.9),
                'vx': np.random.uniform(-0.3, 0.3),  # velocity in x
                'vy': np.random.uniform(-0.3, 0.3),  # velocity in y
                'width': np.random.uniform(0.05, 0.2),
                'height': np.random.uniform(0.05, 0.2),
                'class': np.random.randint(0, 80),  # COCO classes (0-79)
            }
        return objects
    
    def _update_position(self, obj_id, frame):
        """Update object position with some physics"""
        obj = self.objects[obj_id]
        
        # Update position
        obj['x'] += obj['vx'] * self.frame_duration
        obj['y'] += obj['vy'] * self.frame_duration
        
        # Bounce off walls
        if obj['x'] < 0.05 or obj['x'] > 0.95:
            obj['vx'] *= -1
            obj['x'] = np.clip(obj['x'], 0.05, 0.95)
        
        if obj['y'] < 0.05 or obj['y'] > 0.95:
            obj['vy'] *= -1
            obj['y'] = np.clip(obj['y'], 0.05, 0.95)
        
        # Add slight random acceleration
        obj['vx'] += np.random.normal(0, 0.01)
        obj['vy'] += np.random.normal(0, 0.01)
        
        # Add noise to position
        noise_x = np.random.normal(0, 0.002)
        noise_y = np.random.normal(0, 0.002)
        
        return {
            'center_x': obj['x'] + noise_x,
            'center_y': obj['y'] + noise_y,
            'width': obj['width'],
            'height': obj['height'],
            'class': obj['class'],
            'confidence': np.random.uniform(0.7, 0.99)
        }
    
    def generate(self):
        """Generate time series data"""
        data = []
        
        for frame_idx in range(self.num_frames):
            timestamp = datetime.now() + timedelta(seconds=frame_idx * self.frame_duration)
            
            for obj_id in range(self.num_objects):
                detection = self._update_position(obj_id, frame_idx)
                
                data.append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'object_id': obj_id,
                    'time_seconds': frame_idx * self.frame_duration,
                    'center_x': detection['center_x'],
                    'center_y': detection['center_y'],
                    'width': detection['width'],
                    'height': detection['height'],
                    'class': detection['class'],
                    'confidence': detection['confidence']
                })
        
        return pd.DataFrame(data)
    
    def generate_with_features(self):
        """Generate data with derived features (velocity, acceleration)"""
        df = self.generate()
        
        # Calculate velocity for each object
        df = df.sort_values(['object_id', 'frame'])
        
        df['dx'] = df.groupby('object_id')['center_x'].diff()
        df['dy'] = df.groupby('object_id')['center_y'].diff()
        
        # Velocity in normalized coordinates per second
        df['vx'] = df['dx'] / self.frame_duration
        df['vy'] = df['dy'] / self.frame_duration
        
        # Speed (magnitude of velocity)
        df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
        
        # Acceleration
        df['ax'] = df.groupby('object_id')['vx'].diff() / self.frame_duration
        df['ay'] = df.groupby('object_id')['vy'].diff() / self.frame_duration
        
        return df


def save_data(df, output_path):
    """Save dataframe to CSV and JSON formats"""
    # CSV format
    csv_path = output_path.replace('.csv', '') + '.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # JSON format (for each frame)
    json_path = output_path.replace('.csv', '') + '.json'
    json_data = []
    
    for frame_idx in df['frame'].unique():
        frame_data = df[df['frame'] == frame_idx].copy()
        detections = frame_data[[
            'object_id', 'center_x', 'center_y', 'width', 'height', 
            'class', 'confidence'
        ]].to_dict(orient='records')
        
        json_data.append({
            'frame': int(frame_idx),
            'timestamp': frame_data['timestamp'].iloc[0].isoformat(),
            'detections': detections
        })
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    # Generate time series data
    print("Generating YOLO time series data...")
    generator = YOLOTimeSeriesGenerator(num_objects=5, num_frames=500, frame_rate=30)
    
    df = generator.generate_with_features()
    
    # Save data
    output_path = '../data/yolo_timeseries.csv'
    save_data(df, output_path)
    
    print(f"\nGenerated {len(df)} detections from {df['frame'].max()+1} frames")
    print(f"Number of tracked objects: {df['object_id'].nunique()}")
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
