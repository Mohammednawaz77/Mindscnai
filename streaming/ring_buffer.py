"""Ring Buffer for Continuous EEG Data Streaming"""
import numpy as np
from collections import deque
from threading import Lock
from typing import Optional, Tuple

class RingBuffer:
    """Thread-safe ring buffer for continuous signal data"""
    
    def __init__(self, n_channels: int, buffer_seconds: float, sampling_rate: float):
        """
        Initialize ring buffer
        
        Args:
            n_channels: Number of EEG channels
            buffer_seconds: Buffer size in seconds (e.g., 10-30s)
            sampling_rate: Sampling rate in Hz
        """
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.buffer_seconds = buffer_seconds
        self.max_samples = int(buffer_seconds * sampling_rate)
        
        # Thread-safe deque for each channel
        self.buffers = [deque(maxlen=self.max_samples) for _ in range(n_channels)]
        self.lock = Lock()
        self.total_samples_received = 0
        
    def append(self, sample: np.ndarray):
        """
        Append a single multi-channel sample
        
        Args:
            sample: 1D array of shape (n_channels,)
        """
        with self.lock:
            for ch_idx, value in enumerate(sample):
                self.buffers[ch_idx].append(float(value))
            self.total_samples_received += 1
    
    def append_chunk(self, chunk: np.ndarray):
        """
        Append a chunk of samples
        
        Args:
            chunk: 2D array of shape (n_channels, n_samples)
        """
        with self.lock:
            n_samples = chunk.shape[1]
            for ch_idx in range(self.n_channels):
                self.buffers[ch_idx].extend(chunk[ch_idx, :])
            self.total_samples_received += n_samples
    
    def get_latest(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Get latest n samples (or all if n_samples is None)
        
        Args:
            n_samples: Number of samples to retrieve
        
        Returns:
            2D array of shape (n_channels, n_samples)
        """
        with self.lock:
            if n_samples is None:
                n_samples = len(self.buffers[0])
            
            # Get last n_samples from each channel
            data = np.zeros((self.n_channels, min(n_samples, len(self.buffers[0]))))
            for ch_idx in range(self.n_channels):
                buffer_list = list(self.buffers[ch_idx])
                actual_samples = min(n_samples, len(buffer_list))
                data[ch_idx, :actual_samples] = buffer_list[-actual_samples:]
            
            return data
    
    def get_window(self, window_seconds: float, offset_seconds: float = 0) -> Tuple[np.ndarray, bool]:
        """
        Get a time window of data
        
        Args:
            window_seconds: Window size in seconds
            offset_seconds: Offset from current time (0 = most recent)
        
        Returns:
            Tuple of (data array, is_full_window boolean)
        """
        n_samples = int(window_seconds * self.sampling_rate)
        offset_samples = int(offset_seconds * self.sampling_rate)
        
        with self.lock:
            available_samples = len(self.buffers[0])
            
            if available_samples < n_samples:
                # Not enough data yet
                return np.zeros((self.n_channels, n_samples)), False
            
            # Get window
            start_idx = max(0, available_samples - n_samples - offset_samples)
            end_idx = available_samples - offset_samples
            
            data = np.zeros((self.n_channels, end_idx - start_idx))
            for ch_idx in range(self.n_channels):
                buffer_list = list(self.buffers[ch_idx])
                data[ch_idx, :] = buffer_list[start_idx:end_idx]
            
            return data, True
    
    def clear(self):
        """Clear all buffers"""
        with self.lock:
            for buffer in self.buffers:
                buffer.clear()
            self.total_samples_received = 0
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        with self.lock:
            return {
                'n_channels': self.n_channels,
                'buffer_seconds': self.buffer_seconds,
                'max_samples': self.max_samples,
                'current_samples': len(self.buffers[0]),
                'fill_percentage': (len(self.buffers[0]) / self.max_samples) * 100,
                'total_received': self.total_samples_received,
                'sampling_rate': self.sampling_rate
            }
