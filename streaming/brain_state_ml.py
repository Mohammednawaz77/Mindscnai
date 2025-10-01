"""ML Models for Real-Time Brain State Detection"""
import numpy as np
from typing import Dict, Tuple
from scipy import signal as scipy_signal

class BrainStateDetector:
    """Detect cognitive load, focus, and anxiety from EEG signals"""
    
    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate
        
        # Baseline values for calibration (updated during rest period)
        self.baseline = {
            'theta_beta_ratio': 1.0,
            'engagement_index': 1.0,
            'frontal_asymmetry': 0.0,
            'beta_dominance': 0.5
        }
        self.calibrated = False
        
    def compute_band_powers(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute relative band powers using Welch's method
        
        Args:
            data: EEG data (channels × samples)
        
        Returns:
            Dictionary of band powers
        """
        # Average across channels
        avg_signal = np.mean(data, axis=0)
        
        # Compute PSD
        freqs, psd = scipy_signal.welch(avg_signal, fs=self.sampling_rate, 
                                        nperseg=int(self.sampling_rate)*2)
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_powers = {}
        total_power = 0
        
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            power = np.trapz(psd[idx], freqs[idx])
            band_powers[band_name] = power
            total_power += power
        
        # Relative powers
        for band_name in bands.keys():
            band_powers[f'{band_name}_rel'] = (band_powers[band_name] / total_power) * 100
        
        return band_powers
    
    def compute_engagement_metrics(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """
        Compute engagement and attention metrics
        
        Returns:
            Dictionary of engagement metrics
        """
        theta = band_powers.get('theta', 0.01)
        alpha = band_powers.get('alpha', 0.01)
        beta = band_powers.get('beta', 0.01)
        gamma = band_powers.get('gamma', 0.01)
        
        # Theta/Beta ratio (lower = better focus)
        theta_beta_ratio = theta / (beta + 1e-8)
        
        # Engagement index: Beta / (Alpha + Theta)
        engagement_index = beta / (alpha + theta + 1e-8)
        
        # Beta dominance (higher = more alert)
        beta_dominance = beta / (theta + alpha + beta + gamma + 1e-8)
        
        return {
            'theta_beta_ratio': theta_beta_ratio,
            'engagement_index': engagement_index,
            'beta_dominance': beta_dominance
        }
    
    def calibrate(self, rest_data: np.ndarray):
        """
        Calibrate baseline from resting state data
        
        Args:
            rest_data: Resting EEG data (channels × samples), ~60 seconds
        """
        band_powers = self.compute_band_powers(rest_data)
        metrics = self.compute_engagement_metrics(band_powers)
        
        self.baseline = metrics.copy()
        self.calibrated = True
    
    def detect_cognitive_load(self, data: np.ndarray) -> Tuple[float, str]:
        """
        Detect cognitive load level (0-100)
        
        High cognitive load indicators:
        - Higher theta activity
        - Increased frontal theta
        - Higher theta/alpha ratio
        
        Returns:
            Tuple of (load_score, level_description)
        """
        band_powers = self.compute_band_powers(data)
        metrics = self.compute_engagement_metrics(band_powers)
        
        # Higher theta/beta ratio indicates higher cognitive load
        theta_beta = metrics['theta_beta_ratio']
        if self.calibrated:
            theta_beta_normalized = (theta_beta / (self.baseline['theta_beta_ratio'] + 1e-8))
        else:
            theta_beta_normalized = theta_beta
        
        # Convert to 0-100 scale with better scaling
        load_score = min(100, max(0, (theta_beta_normalized - 0.5) * 100))
        
        # Classify level
        if load_score < 30:
            level = "Low"
        elif load_score < 60:
            level = "Moderate"
        else:
            level = "High"
        
        return load_score, level
    
    def detect_focus(self, data: np.ndarray) -> Tuple[float, str]:
        """
        Detect focus/attention level (0-100)
        
        High focus indicators:
        - Lower theta/beta ratio
        - Higher beta activity
        - Lower alpha (eyes open, engaged)
        
        Returns:
            Tuple of (focus_score, level_description)
        """
        band_powers = self.compute_band_powers(data)
        metrics = self.compute_engagement_metrics(band_powers)
        
        # Higher engagement index = better focus
        engagement = metrics['engagement_index']
        beta_dom = metrics['beta_dominance']
        
        if self.calibrated:
            engagement_normalized = engagement / (self.baseline['engagement_index'] + 1e-8)
            beta_dom_normalized = beta_dom / (self.baseline['beta_dominance'] + 1e-8)
        else:
            engagement_normalized = min(2.0, engagement)
            beta_dom_normalized = beta_dom
        
        # Combine engagement and beta dominance with better scaling
        focus_score = min(100, max(0, (engagement_normalized * 30 + beta_dom_normalized * 70 - 20)))
        
        # Classify level
        if focus_score < 30:
            level = "Low"
        elif focus_score < 70:
            level = "Moderate"
        else:
            level = "High"
        
        return focus_score, level
    
    def detect_anxiety(self, data: np.ndarray) -> Tuple[float, str]:
        """
        Detect anxiety level (0-100)
        
        High anxiety indicators:
        - Higher beta activity (especially high beta 20-30 Hz)
        - Lower alpha
        - Higher frontal asymmetry
        
        Returns:
            Tuple of (anxiety_score, level_description)
        """
        band_powers = self.compute_band_powers(data)
        
        beta_rel = band_powers.get('beta_rel', 0)
        alpha_rel = band_powers.get('alpha_rel', 50)
        theta_rel = band_powers.get('theta_rel', 0)
        
        # High beta + low alpha + high theta = potential anxiety
        beta_factor = min(2.0, beta_rel / 20)  # Normalize
        alpha_factor = max(0, 1 - (alpha_rel / 40))  # Lower alpha = higher anxiety
        theta_factor = min(1.5, theta_rel / 15)
        
        anxiety_score = min(100, max(0, (beta_factor * 40 + alpha_factor * 30 + theta_factor * 30 - 30)))
        
        # Classify level
        if anxiety_score < 30:
            level = "Low"
        elif anxiety_score < 60:
            level = "Moderate"
        else:
            level = "High"
        
        return anxiety_score, level
    
    def analyze_window(self, data: np.ndarray) -> Dict:
        """
        Complete brain state analysis for a data window
        
        Args:
            data: EEG data window (channels × samples)
        
        Returns:
            Dictionary with all brain state metrics
        """
        band_powers = self.compute_band_powers(data)
        
        cognitive_load, load_level = self.detect_cognitive_load(data)
        focus, focus_level = self.detect_focus(data)
        anxiety, anxiety_level = self.detect_anxiety(data)
        
        return {
            'cognitive_load': {
                'score': cognitive_load,
                'level': load_level
            },
            'focus': {
                'score': focus,
                'level': focus_level
            },
            'anxiety': {
                'score': anxiety,
                'level': anxiety_level
            },
            'band_powers': band_powers,
            'calibrated': self.calibrated
        }
