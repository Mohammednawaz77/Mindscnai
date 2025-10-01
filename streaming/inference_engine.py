"""Real-Time Inference Engine for Brain State Analysis"""
import time
import numpy as np
from threading import Thread, Lock
from typing import Dict, Optional
from collections import deque

from streaming.ring_buffer import RingBuffer
from streaming.brain_state_ml import BrainStateDetector


class InferenceEngine:
    """Sliding-window inference for real-time brain state analysis"""
    
    def __init__(self, ring_buffer: RingBuffer, 
                 window_seconds: float = 2.0,
                 hop_seconds: float = 0.5,
                 smoothing_alpha: float = 0.3):
        """
        Initialize inference engine
        
        Args:
            ring_buffer: RingBuffer instance with continuous data
            window_seconds: Analysis window size
            hop_seconds: Hop size between windows (overlap = window - hop)
            smoothing_alpha: EWMA smoothing factor (0-1, lower = more smoothing)
        """
        self.ring_buffer = ring_buffer
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.smoothing_alpha = smoothing_alpha
        
        self.detector = BrainStateDetector(sampling_rate=ring_buffer.sampling_rate)
        
        # Inference state
        self.latest_results: Optional[Dict] = None
        self.results_lock = Lock()
        self.is_running = False
        self.inference_thread: Optional[Thread] = None
        
        # Smoothed scores (EWMA)
        self.smoothed_scores = {
            'cognitive_load': 0.0,
            'focus': 0.0,
            'anxiety': 0.0
        }
        
        # History for trend analysis
        self.history = {
            'cognitive_load': deque(maxlen=1200),  # 10 min at 2 Hz
            'focus': deque(maxlen=1200),
            'anxiety': deque(maxlen=1200),
            'timestamps': deque(maxlen=1200)
        }
    
    def calibrate(self):
        """Calibrate detector using current buffer (rest period)"""
        data, is_full = self.ring_buffer.get_window(window_seconds=30.0)
        if is_full:
            self.detector.calibrate(data)
            return True
        return False
    
    def start(self):
        """Start inference engine"""
        if self.is_running:
            return
        
        self.is_running = True
        self.inference_thread = Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
    
    def stop(self):
        """Stop inference engine"""
        self.is_running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
    
    def _inference_loop(self):
        """Main inference loop"""
        while self.is_running:
            try:
                # Get data window
                data, is_full = self.ring_buffer.get_window(self.window_seconds)
                
                if not is_full:
                    # Not enough data yet
                    time.sleep(0.1)
                    continue
                
                # Run analysis
                results = self.detector.analyze_window(data)
                
                # Apply EWMA smoothing
                self.smoothed_scores['cognitive_load'] = (
                    self.smoothing_alpha * results['cognitive_load']['score'] +
                    (1 - self.smoothing_alpha) * self.smoothed_scores['cognitive_load']
                )
                self.smoothed_scores['focus'] = (
                    self.smoothing_alpha * results['focus']['score'] +
                    (1 - self.smoothing_alpha) * self.smoothed_scores['focus']
                )
                self.smoothed_scores['anxiety'] = (
                    self.smoothing_alpha * results['anxiety']['score'] +
                    (1 - self.smoothing_alpha) * self.smoothed_scores['anxiety']
                )
                
                # Update results with smoothed scores
                results['cognitive_load']['smoothed_score'] = self.smoothed_scores['cognitive_load']
                results['focus']['smoothed_score'] = self.smoothed_scores['focus']
                results['anxiety']['smoothed_score'] = self.smoothed_scores['anxiety']
                
                # Store in history
                timestamp = time.time()
                self.history['cognitive_load'].append(self.smoothed_scores['cognitive_load'])
                self.history['focus'].append(self.smoothed_scores['focus'])
                self.history['anxiety'].append(self.smoothed_scores['anxiety'])
                self.history['timestamps'].append(timestamp)
                
                # Update latest results
                with self.results_lock:
                    self.latest_results = results
                
                # Sleep until next window
                time.sleep(self.hop_seconds)
                
            except Exception as e:
                print(f"Inference error: {e}")
                time.sleep(0.5)
    
    def get_latest_results(self) -> Optional[Dict]:
        """Get latest inference results"""
        with self.results_lock:
            return self.latest_results
    
    def get_history(self, minutes: int = 10) -> Dict:
        """Get historical data for trend visualization"""
        max_points = int((minutes * 60) / self.hop_seconds)
        
        with self.results_lock:
            return {
                'cognitive_load': list(self.history['cognitive_load'])[-max_points:],
                'focus': list(self.history['focus'])[-max_points:],
                'anxiety': list(self.history['anxiety'])[-max_points:],
                'timestamps': list(self.history['timestamps'])[-max_points:]
            }
    
    def get_stats(self) -> Dict:
        """Get inference engine statistics"""
        with self.results_lock:
            return {
                'window_seconds': self.window_seconds,
                'hop_seconds': self.hop_seconds,
                'smoothing_alpha': self.smoothing_alpha,
                'is_running': self.is_running,
                'calibrated': self.detector.calibrated,
                'history_length': len(self.history['cognitive_load']),
                'latest_update': self.history['timestamps'][-1] if self.history['timestamps'] else None
            }
