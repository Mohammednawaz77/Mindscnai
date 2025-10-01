import numpy as np
from typing import Tuple, List
import os
from datetime import datetime

class Helpers:
    """Utility helper functions"""
    
    @staticmethod
    def create_dummy_labels(n_samples: int) -> np.ndarray:
        """Create dummy binary labels for classification"""
        # Create balanced labels
        labels = np.zeros(n_samples, dtype=int)
        labels[n_samples//2:] = 1
        return labels
    
    @staticmethod
    def split_data(data: np.ndarray, labels: np.ndarray, 
                   test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets"""
        n_samples = len(data)
        n_test = int(n_samples * test_size)
        
        # Random shuffle
        indices = np.random.permutation(n_samples)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = data[train_indices]
        X_test = data[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def ensure_directory(path: str):
        """Ensure directory exists"""
        if not os.path.exists(path):
            os.makedirs(path)
    
    @staticmethod
    def format_timestamp(timestamp: str | None = None) -> str:
        """Format timestamp for display"""
        if timestamp is None:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return timestamp
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to readable string"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
