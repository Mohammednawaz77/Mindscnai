"""EDF File Replay for Simulating Live Streaming"""
import time
import numpy as np
from threading import Thread, Event
from typing import Callable, Optional
from pathlib import Path

from preprocessing.eeg_processor import EEGProcessor
from preprocessing.channel_selector import ChannelSelector


class EDFReplayStreamer:
    """Stream EDF file data to simulate live EEG acquisition"""
    
    def __init__(self, edf_file_path: str, target_channels: int = 20, 
                 playback_speed: float = 1.0):
        """
        Initialize EDF replay streamer
        
        Args:
            edf_file_path: Path to EDF file
            target_channels: Number of channels to stream
            playback_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)
        """
        self.edf_file_path = edf_file_path
        self.target_channels = target_channels
        self.playback_speed = playback_speed
        
        self.processor = EEGProcessor()
        self.selector = ChannelSelector(target_channels=target_channels)
        
        # Load and preprocess data
        self.data, self.channel_names, self.sampling_rate = self.processor.read_edf_file(edf_file_path)
        preprocessed = self.processor.preprocess_signals(self.data)
        self.selected_data, self.selected_channels, _ = self.selector.select_optimal_channels(
            preprocessed, self.channel_names, method='names'
        )
        
        self.n_channels, self.n_samples = self.selected_data.shape
        
        # Streaming state
        self.current_sample = 0
        self.is_streaming = False
        self.stop_event = Event()
        self.stream_thread: Optional[Thread] = None
        self.callback: Optional[Callable] = None
        
    def start_stream(self, callback: Callable[[np.ndarray, float], None], 
                    loop: bool = True):
        """
        Start streaming data
        
        Args:
            callback: Function to call with each sample chunk (data, timestamp)
            loop: Loop the file when it ends
        """
        if self.is_streaming:
            return
        
        self.callback = callback
        self.is_streaming = True
        self.stop_event.clear()
        self.current_sample = 0
        
        self.stream_thread = Thread(target=self._stream_loop, args=(loop,), daemon=True)
        self.stream_thread.start()
    
    def _stream_loop(self, loop: bool):
        """Internal streaming loop"""
        chunk_size = 1  # Stream one sample at a time for smoothness
        sleep_time = (chunk_size / self.sampling_rate) / self.playback_speed
        
        while self.is_streaming and not self.stop_event.is_set():
            # Get next chunk
            end_sample = min(self.current_sample + chunk_size, self.n_samples)
            chunk = self.selected_data[:, self.current_sample:end_sample]
            
            if chunk.shape[1] == 0:
                if loop:
                    # Restart from beginning
                    self.current_sample = 0
                    continue
                else:
                    # End of file
                    self.is_streaming = False
                    break
            
            # Send chunk to callback
            timestamp = time.time()
            if self.callback:
                try:
                    if chunk.shape[1] == 1:
                        # Single sample
                        self.callback(chunk[:, 0], timestamp)
                    else:
                        # Multiple samples
                        for i in range(chunk.shape[1]):
                            self.callback(chunk[:, i], timestamp)
                except Exception as e:
                    print(f"Callback error: {e}")
            
            self.current_sample = end_sample
            
            # Sleep to maintain real-time rate
            time.sleep(sleep_time)
    
    def stop_stream(self):
        """Stop streaming"""
        self.is_streaming = False
        self.stop_event.set()
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
    
    def get_info(self) -> dict:
        """Get stream information"""
        return {
            'file': Path(self.edf_file_path).name,
            'n_channels': self.n_channels,
            'sampling_rate': self.sampling_rate,
            'total_samples': self.n_samples,
            'duration_seconds': self.n_samples / self.sampling_rate,
            'current_sample': self.current_sample,
            'progress_percent': (self.current_sample / self.n_samples) * 100,
            'is_streaming': self.is_streaming,
            'playback_speed': self.playback_speed
        }
