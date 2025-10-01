"""Advanced Signal Processing for EEG with Artifact Removal and Adaptive Filtering"""
import numpy as np
from scipy import signal
from scipy.stats import zscore
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedSignalProcessor:
    """Advanced signal processing with artifact removal and adaptive filtering"""
    
    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate
    
    def remove_artifacts_ica_simple(self, data: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Simple ICA-based artifact removal using FastICA approximation
        
        Args:
            data: EEG data (channels × samples)
            n_components: Number of ICA components (None = all channels)
        
        Returns:
            Cleaned EEG data
        """
        n_channels, n_samples = data.shape
        
        if n_components is None:
            n_components = min(n_channels, 10)  # Limit components for efficiency
        
        # Center the data
        data_centered = data - data.mean(axis=1, keepdims=True)
        
        # Whitening (PCA-based)
        cov_matrix = np.cov(data_centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top components
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
        
        # Whitening matrix
        whitening_matrix = eigenvectors.T / np.sqrt(eigenvalues[:, np.newaxis])
        whitened_data = whitening_matrix @ data_centered
        
        # Simple FastICA approximation with fixed iterations
        W = np.random.randn(n_components, n_components)
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        
        for _ in range(20):  # Limited iterations
            W_new = (whitened_data @ np.tanh(W.T @ whitened_data).T) / n_samples
            W_new -= W_new @ W.T @ W
            W_new = W_new / np.linalg.norm(W_new, axis=1, keepdims=True)
            
            if np.allclose(np.abs(W @ W_new.T), np.eye(n_components), atol=1e-3):
                break
            W = W_new
        
        # Get independent components
        independent_components = W @ whitened_data
        
        # Automatic artifact detection based on kurtosis and variance
        artifact_threshold_kurt = 5.0
        artifact_threshold_var = 3.0
        
        kurt = self._kurtosis(independent_components, axis=1)
        variance = np.var(independent_components, axis=1)
        var_zscore = np.abs(zscore(variance))  # type: ignore
        
        # Mark components with high kurtosis or extreme variance as artifacts
        artifact_mask = (np.abs(kurt) > artifact_threshold_kurt) | (var_zscore > artifact_threshold_var)
        
        # Zero out artifact components
        independent_components[artifact_mask, :] = 0
        
        # Reconstruct cleaned data
        dewhitening_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        cleaned_data = dewhitening_matrix @ np.linalg.pinv(W) @ independent_components
        
        return cleaned_data + data.mean(axis=1, keepdims=True)
    
    def _kurtosis(self, data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute kurtosis along axis"""
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        return np.mean(normalized ** 4, axis=axis) - 3
    
    def adaptive_noise_filter(self, data: np.ndarray, method: str = 'wiener') -> np.ndarray:
        """
        Adaptive noise filtering
        
        Args:
            data: EEG data (channels × samples)
            method: 'wiener' or 'wavelet'
        
        Returns:
            Filtered EEG data
        """
        if method == 'wiener':
            return self._wiener_filter(data)
        elif method == 'wavelet':
            return self._wavelet_denoising(data)
        else:
            return data
    
    def _wiener_filter(self, data: np.ndarray, noise_power: Optional[float] = None) -> np.ndarray:
        """Apply Wiener filter for noise reduction"""
        n_channels, n_samples = data.shape
        filtered_data = np.zeros_like(data)
        
        for ch in range(n_channels):
            channel_data = data[ch, :]
            
            # Estimate noise power from high-frequency components
            if noise_power is None:
                # High-pass filter to estimate noise
                nyq = self.sampling_rate / 2
                high_cutoff = 40.0
                b, a = signal.butter(4, high_cutoff / nyq, btype='high')
                noise_estimate = signal.filtfilt(b, a, channel_data)
                est_noise_power = np.var(noise_estimate)
            else:
                est_noise_power = noise_power
            
            # Signal power
            signal_power = np.var(channel_data)
            
            # Wiener gain
            gain = max(0, 1 - est_noise_power / (signal_power + 1e-8))
            
            # Apply adaptive filtering
            filtered_data[ch, :] = channel_data * gain
        
        return filtered_data
    
    def _wavelet_denoising(self, data: np.ndarray) -> np.ndarray:
        """Simple wavelet-based denoising using soft thresholding"""
        n_channels, n_samples = data.shape
        filtered_data = np.zeros_like(data)
        
        for ch in range(n_channels):
            channel_data = data[ch, :]
            
            # Simple wavelet decomposition approximation using DWT
            # Approximate with multi-resolution filtering
            coeffs = []
            temp_data = channel_data.copy()
            
            # 3-level decomposition
            for level in range(3):
                # Low-pass (approximation)
                b, a = signal.butter(2, 0.5 / (2 ** level), btype='low')
                approx = signal.filtfilt(b, a, temp_data)
                
                # Detail (difference)
                detail = temp_data - approx
                
                # Soft thresholding
                threshold = np.sqrt(2 * np.log(len(detail))) * np.std(detail)
                detail_thresholded = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
                
                coeffs.append(detail_thresholded)
                temp_data = approx
            
            # Reconstruct
            reconstructed = temp_data
            for detail in reversed(coeffs):
                reconstructed = reconstructed + detail
            
            filtered_data[ch, :] = reconstructed
        
        return filtered_data
    
    def detect_bad_channels(self, data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, list]:
        """
        Detect bad channels based on amplitude and variance
        
        Args:
            data: EEG data (channels × samples)
            threshold: Z-score threshold for bad channel detection
        
        Returns:
            Tuple of (cleaned_data, bad_channel_indices)
        """
        n_channels, n_samples = data.shape
        
        # Compute channel statistics
        channel_variance = np.var(data, axis=1)
        channel_max = np.max(np.abs(data), axis=1)
        
        # Z-score based detection
        var_zscore = np.abs(zscore(channel_variance))  # type: ignore
        max_zscore = np.abs(zscore(channel_max))  # type: ignore
        
        # Bad channels: extreme variance or amplitude
        bad_channels = np.where((var_zscore > threshold) | (max_zscore > threshold))[0]
        
        # Interpolate bad channels (average of neighbors)
        cleaned_data = data.copy()
        for bad_ch in bad_channels:
            if bad_ch > 0 and bad_ch < n_channels - 1:
                cleaned_data[bad_ch, :] = (data[bad_ch - 1, :] + data[bad_ch + 1, :]) / 2
            elif bad_ch == 0 and n_channels > 1:
                cleaned_data[bad_ch, :] = data[bad_ch + 1, :]
            elif bad_ch == n_channels - 1 and n_channels > 1:
                cleaned_data[bad_ch, :] = data[bad_ch - 1, :]
        
        return cleaned_data, bad_channels.tolist()
    
    def apply_advanced_preprocessing(self, data: np.ndarray, 
                                     remove_artifacts: bool = True,
                                     adaptive_filter: bool = True,
                                     detect_bad: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Apply full advanced preprocessing pipeline
        
        Args:
            data: EEG data (channels × samples)
            remove_artifacts: Apply ICA-based artifact removal
            adaptive_filter: Apply adaptive noise filtering
            detect_bad: Detect and interpolate bad channels
        
        Returns:
            Tuple of (processed_data, processing_info)
        """
        processing_info = {
            'artifact_removal': remove_artifacts,
            'adaptive_filtering': adaptive_filter,
            'bad_channel_detection': detect_bad,
            'bad_channels': []
        }
        
        processed_data = data.copy()
        
        # Step 1: Bad channel detection
        if detect_bad:
            processed_data, bad_channels = self.detect_bad_channels(processed_data)
            processing_info['bad_channels'] = bad_channels
        
        # Step 2: Artifact removal
        if remove_artifacts:
            processed_data = self.remove_artifacts_ica_simple(processed_data)
        
        # Step 3: Adaptive noise filtering
        if adaptive_filter:
            processed_data = self.adaptive_noise_filter(processed_data, method='wiener')
        
        return processed_data, processing_info
