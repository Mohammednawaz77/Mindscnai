import numpy as np
from typing import List, Tuple
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr

class ChannelSelector:
    """Select optimal 20 channels from 64-channel EEG data"""
    
    def __init__(self, target_channels: int = 20):
        self.target_channels = target_channels
        
        # Standard 10-20 system priority channels for BCI
        self.priority_channels = [
            'Fz', 'Cz', 'Pz', 'C3', 'C4',  # Motor cortex
            'F3', 'F4', 'P3', 'P4',  # Frontal and parietal
            'O1', 'O2',  # Visual cortex
            'T7', 'T8',  # Temporal
            'Fp1', 'Fp2',  # Frontal pole
            'FC1', 'FC2', 'CP1', 'CP2',  # Central
            'PO7', 'PO8'  # Parieto-occipital
        ]
    
    def select_channels_by_names(self, channel_names: List[str]) -> Tuple[List[int], List[str]]:
        """Select channels based on standard 10-20 system names"""
        selected_indices = []
        selected_names = []
        
        # First, try to match priority channels
        for priority in self.priority_channels:
            for i, name in enumerate(channel_names):
                if priority.lower() in name.lower() and i not in selected_indices:
                    selected_indices.append(i)
                    selected_names.append(name)
                    break
        
        # If we don't have enough channels, add more based on position
        if len(selected_indices) < self.target_channels:
            for i, name in enumerate(channel_names):
                if i not in selected_indices:
                    selected_indices.append(i)
                    selected_names.append(name)
                    if len(selected_indices) >= self.target_channels:
                        break
        
        # Take only the target number
        selected_indices = selected_indices[:self.target_channels]
        selected_names = selected_names[:self.target_channels]
        
        return selected_indices, selected_names
    
    def select_channels_by_variance(self, data: np.ndarray) -> List[int]:
        """Select channels with highest variance"""
        variances = np.var(data, axis=1)
        selected_indices = np.argsort(variances)[-self.target_channels:]
        return sorted(selected_indices.tolist())
    
    def select_channels_by_mutual_info(self, data: np.ndarray, 
                                      labels: np.ndarray = None) -> List[int]:
        """Select channels based on mutual information with labels"""
        
        if labels is None:
            # Use variance-based selection if no labels
            return self.select_channels_by_variance(data)
        
        # Compute mutual information for each channel
        mi_scores = []
        for i in range(data.shape[0]):
            # Use mean of each channel as feature
            channel_feature = np.mean(data[i].reshape(-1, 1), axis=1)
            mi = mutual_info_classif(channel_feature.reshape(-1, 1), labels)
            mi_scores.append(mi[0])
        
        mi_scores = np.array(mi_scores)
        selected_indices = np.argsort(mi_scores)[-self.target_channels:]
        return sorted(selected_indices.tolist())
    
    def select_channels_by_correlation(self, data: np.ndarray) -> List[int]:
        """Select channels with low inter-channel correlation"""
        n_channels = data.shape[0]
        
        # Compute correlation matrix
        corr_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = pearsonr(data[i], data[j])
                    corr_matrix[i, j] = abs(corr)
                    corr_matrix[j, i] = abs(corr)
        
        # Select channels with lowest average correlation
        avg_corr = np.mean(corr_matrix, axis=1)
        selected_indices = np.argsort(avg_corr)[:self.target_channels]
        return sorted(selected_indices.tolist())
    
    def select_optimal_channels(self, data: np.ndarray, 
                               channel_names: List[str],
                               method: str = 'names') -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Select optimal channels using specified method
        
        Args:
            data: EEG data (channels x samples)
            channel_names: List of channel names
            method: 'names', 'variance', 'correlation'
        
        Returns:
            selected_data, selected_indices, selected_names
        """
        
        if method == 'names':
            selected_indices, selected_names = self.select_channels_by_names(channel_names)
        elif method == 'variance':
            selected_indices = self.select_channels_by_variance(data)
            selected_names = [channel_names[i] for i in selected_indices]
        elif method == 'correlation':
            selected_indices = self.select_channels_by_correlation(data)
            selected_names = [channel_names[i] for i in selected_indices]
        else:
            # Default to names
            selected_indices, selected_names = self.select_channels_by_names(channel_names)
        
        selected_data = data[selected_indices, :]
        
        return selected_data, selected_indices, selected_names
