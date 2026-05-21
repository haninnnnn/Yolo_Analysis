"""
Real-time YOLO detection processor.
Takes live YOLO detections and predicts next positions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class RealTimeYOLOPredictor:
    """Process real-time YOLO detections and predict movements"""
    
    def __init__(self, model_results_file=None, look_back=10):
        """
        Args:
            model_results_file: Path to saved prediction_results.json
            look_back: Number of past frames to maintain
        """
        self.look_back = look_back
        self.tracking_buffer = {}  # Buffer for each object
        self.predictions = {}
        self.formulas = None
        
        if model_results_file:
            with open(model_results_file, 'r') as f:
                self.model_results = json.load(f)
                if 'formulas' in self.model_results:
                    self.formulas = self.model_results['formulas']
    
    def process_frame(self, frame_id, detections):
        """
        Process a frame of YOLO detections.
        
        Args:
            frame_id: Frame number/ID
            detections: List of dicts with keys:
                       [object_id, center_x, center_y, width, height, class, confidence]
        
        Returns:
            Predictions for each detected object
        """
        predictions = {}
        timestamp = datetime.now()
        
        for det in detections:
            obj_id = det['object_id']
            
            # Initialize buffer for new objects
            if obj_id not in self.tracking_buffer:
                self.tracking_buffer[obj_id] = []
            
            # Store detection
            track_point = {
                'frame': frame_id,
                'timestamp': timestamp,
                'center_x': det['center_x'],
                'center_y': det['center_y'],
                'width': det['width'],
                'height': det['height'],
                'confidence': det['confidence'],
            }
            
            # Calculate velocity if we have history
            if len(self.tracking_buffer[obj_id]) > 0:
                last_point = self.tracking_buffer[obj_id][-1]
                dt = (timestamp - last_point['timestamp']).total_seconds()
                
                if dt > 0:
                    track_point['dx'] = det['center_x'] - last_point['center_x']
                    track_point['dy'] = det['center_y'] - last_point['center_y']
                    track_point['vx'] = track_point['dx'] / dt
                    track_point['vy'] = track_point['dy'] / dt
                    track_point['speed'] = np.sqrt(track_point['vx']**2 + track_point['vy']**2)
            
            # Add to buffer
            self.tracking_buffer[obj_id].append(track_point)
            
            # Keep only look_back frames
            if len(self.tracking_buffer[obj_id]) > self.look_back:
                self.tracking_buffer[obj_id].pop(0)
            
            # Generate prediction if we have enough history
            if len(self.tracking_buffer[obj_id]) > 2:
                pred = self.predict_next_position(obj_id)
                if pred:
                    predictions[obj_id] = pred
        
        return predictions
    
    def predict_next_position(self, obj_id, steps=1):
        """
        Predict next position using linear formula.
        
        Args:
            obj_id: Object ID
            steps: Number of frames ahead to predict
        
        Returns:
            Predicted position(s)
        """
        history = self.tracking_buffer[obj_id]
        
        if len(history) < 2:
            return None
        
        # Use most recent velocity
        recent = history[-1]
        
        if 'vx' not in recent or 'vy' not in recent:
            return None
        
        vx = recent['vx']
        vy = recent['vy']
        current_x = recent['center_x']
        current_y = recent['center_y']
        
        # Simple constant velocity model
        predictions = []
        for step in range(steps):
            next_x = current_x + vx * step
            next_y = current_y + vy * step
            
            predictions.append({
                'step': step + 1,
                'predicted_x': float(next_x),
                'predicted_y': float(next_y),
                'confidence': float(recent['confidence']),
                'velocity_x': float(vx),
                'velocity_y': float(vy),
                'speed': float(np.sqrt(vx**2 + vy**2))
            })
        
        return predictions[0] if steps == 1 else predictions
    
    def get_object_trajectory(self, obj_id):
        """Get historical trajectory for an object"""
        if obj_id not in self.tracking_buffer:
            return None
        
        history = self.tracking_buffer[obj_id]
        return {
            'object_id': obj_id,
            'num_frames': len(history),
            'positions': [
                {
                    'frame': point['frame'],
                    'x': point['center_x'],
                    'y': point['center_y'],
                    'confidence': point['confidence']
                }
                for point in history
            ]
        }
    
    def get_tracking_statistics(self):
        """Get statistics on current tracking"""
        stats = {
            'total_objects_tracked': len(self.tracking_buffer),
            'objects': {}
        }
        
        for obj_id, history in self.tracking_buffer.items():
            if len(history) > 1:
                velocities = [
                    np.sqrt(p.get('vx', 0)**2 + p.get('vy', 0)**2)
                    for p in history[1:]
                    if 'vx' in p
                ]
                
                stats['objects'][obj_id] = {
                    'frames_tracked': len(history),
                    'avg_velocity': float(np.mean(velocities)) if velocities else 0,
                    'max_velocity': float(np.max(velocities)) if velocities else 0,
                    'confidence': float(history[-1].get('confidence', 0))
                }
        
        return stats
    
    def export_tracking_data(self, output_file):
        """Export all tracking data to file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'objects': {
                str(obj_id): self.get_object_trajectory(obj_id)
                for obj_id in self.tracking_buffer.keys()
            },
            'statistics': self.get_tracking_statistics()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Tracking data exported to {output_file}")


# Example usage
if __name__ == "__main__":
    # Initialize real-time predictor
    predictor = RealTimeYOLOPredictor(look_back=10)
    
    # Simulate real-time frame processing
    print("Simulating real-time YOLO detection processing...")
    
    # Frame 1: 2 objects detected
    frame_1_detections = [
        {'object_id': 0, 'center_x': 0.5, 'center_y': 0.5, 'width': 0.1, 'height': 0.15, 'confidence': 0.95},
        {'object_id': 1, 'center_x': 0.3, 'center_y': 0.7, 'width': 0.12, 'height': 0.14, 'confidence': 0.92},
    ]
    preds_1 = predictor.process_frame(frame_id=1, detections=frame_1_detections)
    print(f"Frame 1 predictions: {preds_1}")
    
    # Frame 2: Same objects with slight movement
    frame_2_detections = [
        {'object_id': 0, 'center_x': 0.502, 'center_y': 0.505, 'width': 0.1, 'height': 0.15, 'confidence': 0.96},
        {'object_id': 1, 'center_x': 0.31, 'center_y': 0.71, 'width': 0.12, 'height': 0.14, 'confidence': 0.93},
    ]
    preds_2 = predictor.process_frame(frame_id=2, detections=frame_2_detections)
    print(f"Frame 2 predictions: {preds_2}")
    
    # Frame 3
    frame_3_detections = [
        {'object_id': 0, 'center_x': 0.504, 'center_y': 0.51, 'width': 0.1, 'height': 0.15, 'confidence': 0.94},
        {'object_id': 1, 'center_x': 0.32, 'center_y': 0.72, 'width': 0.12, 'height': 0.14, 'confidence': 0.91},
    ]
    preds_3 = predictor.process_frame(frame_id=3, detections=frame_3_detections)
    print(f"Frame 3 predictions: {preds_3}")
    
    # Get tracking statistics
    print("\nTracking Statistics:")
    stats = predictor.get_tracking_statistics()
    for obj_id, obj_stats in stats['objects'].items():
        print(f"  Object {obj_id}: avg velocity = {obj_stats['avg_velocity']:.4f}, confidence = {obj_stats['confidence']:.3f}")
    
    # Export tracking data
    predictor.export_tracking_data('tracking_output.json')
    print("\n✓ Real-time processing example complete!")
