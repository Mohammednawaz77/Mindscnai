import numpy as np
import mne
from scipy import signal
from typing import Tuple, List
import pyedflib

class EEGProcessor:
    """Process EEG data from EDF files"""
    
    def __init__(self):
        self.sampling_rate = 256  # Default sampling rate
        self.channels_to_select = 20
    
    def read_edf_file(self, file_path: str) -> Tuple[np.ndarray, List[str], float]:
        """Read EDF file and extract data"""
        try:
            # Read using pyedflib
            edf_file = pyedflib.EdfReader(file_path)
            n_channels = edf_file.signals_in_file
            signal_labels = edf_file.getSignalLabels()
            
            # Get sampling frequency
            self.sampling_rate = edf_file.getSampleFrequency(0)
            
            # Read all signals
            signals = []
            for i in range(n_channels):
                signals.append(edf_file.readSignal(i))
            
            edf_file.close()
            
            # Convert to numpy array
            data = np.array(signals)
            
            return data, signal_labels, self.sampling_rate
            
        except Exception as e:
            raise Exception(f"Error reading EDF file: {str(e)}")
    
    def preprocess_signals(self, data: np.ndarray) -> np.ndarray:
        """Preprocess EEG signals"""
        
        # 1. Bandpass filter (0.5-50 Hz)
        filtered_data = self._bandpass_filter(data, 0.5, 50.0, self.sampling_rate)
        
        # 2. Notch filter (50 Hz for power line noise)
        notched_data = self._notch_filter(filtered_data, 50.0, self.sampling_rate)
        
        # 3. Normalize
        normalized_data = self._normalize(notched_data)
        
        return normalized_data
    
    def _bandpass_filter(self, data: np.ndarray, lowcut: float, 
                        highcut: float, fs: float, order: int = 4) -> np.ndarray:
        """Apply bandpass filter"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = signal.filtfilt(b, a, data[i])
        
        return filtered
    
    def _notch_filter(self, data: np.ndarray, freq: float, 
                     fs: float, quality: float = 30.0) -> np.ndarray:
        """Apply notch filter"""
        b, a = signal.iirnotch(freq, quality, fs)
        
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = signal.filtfilt(b, a, data[i])
        
        return filtered
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize signals using z-score"""
        normalized = np.zeros_like(data)
        for i in range(data.shape[0]):
            mean = np.mean(data[i])
            std = np.std(data[i])
            if std > 0:
                normalized[i] = (data[i] - mean) / std
            else:
                normalized[i] = data[i] - mean
        
        return normalized
    
    def extract_epochs(self, data: np.ndarray, epoch_length: float = 2.0) -> np.ndarray:
        """Extract epochs from continuous data"""
        samples_per_epoch = int(epoch_length * self.sampling_rate)
        n_channels, n_samples = data.shape
        n_epochs = n_samples // samples_per_epoch
        
        epochs = []
        for i in range(n_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            epoch = data[:, start:end]
            epochs.append(epoch)
        
        return np.array(epochs)
    
    def downsample(self, data: np.ndarray, target_rate: int = 128) -> np.ndarray:
        """Downsample signals to target rate"""
        if target_rate >= self.sampling_rate:
            return data
        
        downsample_factor = int(self.sampling_rate / target_rate)
        downsampled = signal.decimate(data, downsample_factor, axis=1)
        self.sampling_rate = target_rate
        
        return downsampled
