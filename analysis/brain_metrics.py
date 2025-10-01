import numpy as np
from scipy import signal
from scipy.integrate import simpson
from typing import Dict, Tuple

class BrainMetricsAnalyzer:
    """Analyze brain metrics from EEG signals"""
    
    def __init__(self, sampling_rate: float = 256):
        self.sampling_rate = sampling_rate
        
        # Frequency bands (Hz)
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def compute_psd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Power Spectral Density"""
        freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=int(self.sampling_rate)*2)
        return freqs, psd
    
    def compute_band_power(self, data: np.ndarray, band: Tuple[float, float]) -> float:
        """Compute power in a specific frequency band"""
        freqs, psd = self.compute_psd(data)
        
        # Find indices of frequencies in band
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        
        # Compute power using Simpson's rule
        band_power = simpson(psd[idx_band], x=freqs[idx_band])
        
        return band_power
    
    def compute_all_band_powers(self, data: np.ndarray) -> Dict[str, float]:
        """Compute power in all frequency bands"""
        band_powers = {}
        
        for band_name, band_range in self.bands.items():
            band_powers[band_name] = self.compute_band_power(data, band_range)
        
        return band_powers
    
    def compute_relative_band_powers(self, data: np.ndarray) -> Dict[str, float]:
        """Compute relative power in all frequency bands"""
        band_powers = self.compute_all_band_powers(data)
        total_power = sum(band_powers.values())
        
        relative_powers = {}
        for band_name, power in band_powers.items():
            relative_powers[f'{band_name}_relative'] = (power / total_power) * 100 if total_power > 0 else 0
        
        return relative_powers
    
    def compute_channel_metrics(self, channel_data: np.ndarray) -> Dict:
        """Compute comprehensive metrics for a single channel"""
        # Band powers
        band_powers = self.compute_all_band_powers(channel_data)
        relative_powers = self.compute_relative_band_powers(channel_data)
        
        # Statistical features
        metrics = {
            'mean': float(np.mean(channel_data)),
            'std': float(np.std(channel_data)),
            'variance': float(np.var(channel_data)),
            'min': float(np.min(channel_data)),
            'max': float(np.max(channel_data)),
            'range': float(np.ptp(channel_data)),
            'rms': float(np.sqrt(np.mean(channel_data**2)))
        }
        
        # Add band powers
        metrics.update(band_powers)
        metrics.update(relative_powers)
        
        return metrics
    
    def compute_multi_channel_metrics(self, data: np.ndarray) -> Dict:
        """Compute metrics for all channels"""
        n_channels = data.shape[0]
        
        # Average across all channels
        avg_data = np.mean(data, axis=0)
        avg_metrics = self.compute_channel_metrics(avg_data)
        
        # Per-channel metrics
        channel_metrics = []
        for i in range(n_channels):
            channel_metrics.append(self.compute_channel_metrics(data[i]))
        
        # Aggregate metrics
        all_metrics = {
            'average': avg_metrics,
            'channels': channel_metrics,
            'n_channels': n_channels
        }
        
        return all_metrics
    
    def compute_brain_state(self, data: np.ndarray) -> str:
        """Determine brain state based on dominant frequency band"""
        band_powers = self.compute_all_band_powers(np.mean(data, axis=0))
        
        # Find dominant band
        dominant_band = max(band_powers.items(), key=lambda x: x[1])[0]
        
        # Map to brain state
        state_mapping = {
            'delta': 'Deep Sleep',
            'theta': 'Drowsy/Meditative',
            'alpha': 'Relaxed/Wakeful',
            'beta': 'Active/Alert',
            'gamma': 'Highly Focused'
        }
        
        return state_mapping.get(dominant_band, 'Unknown')
    
    def compute_asymmetry_index(self, left_channel: np.ndarray, 
                               right_channel: np.ndarray, band: str = 'alpha') -> float:
        """Compute hemispheric asymmetry index"""
        band_range = self.bands[band]
        
        left_power = self.compute_band_power(left_channel, band_range)
        right_power = self.compute_band_power(right_channel, band_range)
        
        # Asymmetry index
        if left_power + right_power > 0:
            asymmetry = (right_power - left_power) / (right_power + left_power)
        else:
            asymmetry = 0
        
        return asymmetry
    
    def get_comprehensive_report(self, data: np.ndarray, channel_names: list | None = None) -> Dict:
        """Generate comprehensive brain metrics report"""
        avg_data = np.mean(data, axis=0)
        
        # Basic metrics
        band_powers = self.compute_all_band_powers(avg_data)
        relative_powers = self.compute_relative_band_powers(avg_data)
        
        # Brain state
        brain_state = self.compute_brain_state(data)
        
        # Total power
        total_power = sum(band_powers.values())
        
        report = {
            'brain_state': brain_state,
            'total_power': total_power,
            'alpha_power': band_powers['alpha'],
            'beta_power': band_powers['beta'],
            'theta_power': band_powers['theta'],
            'delta_power': band_powers['delta'],
            'gamma_power': band_powers['gamma'],
            'alpha_relative': relative_powers['alpha_relative'],
            'beta_relative': relative_powers['beta_relative'],
            'theta_relative': relative_powers['theta_relative'],
            'delta_relative': relative_powers['delta_relative'],
            'gamma_relative': relative_powers['gamma_relative'],
            'n_channels': data.shape[0]
        }
        
        if channel_names:
            report['channel_names'] = channel_names
        
        return report
