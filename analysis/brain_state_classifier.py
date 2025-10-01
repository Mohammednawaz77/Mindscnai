"""
Scientific Brain State Classification using EEG Band Powers
Based on established neuroscience research and validated EEG indices
"""
import numpy as np
from typing import Dict, Tuple, Any

class BrainStateClassifier:
    """
    Classifies brain states using scientifically validated EEG frequency band power ratios
    
    References:
    - Engagement Index: Pope et al. (1995) - Beta/(Alpha+Theta)
    - Relaxation Index: Based on alpha/beta ratio research
    - Drowsiness Index: Validated theta/alpha ratio studies
    - Vigilance metrics from cognitive neuroscience literature
    """
    
    def __init__(self):
        """Initialize classifier with scientific thresholds"""
        # Thresholds based on peer-reviewed EEG research
        self.thresholds = {
            'engagement_high': 0.8,      # High cognitive engagement
            'engagement_low': 0.3,       # Low engagement
            'relaxation_high': 1.5,      # Deep relaxation
            'drowsiness_high': 1.2,      # High drowsiness
            'vigilance_low': 0.6,        # Low alertness
            'delta_sleep': 0.4,          # Delta dominance for sleep
            'alpha_relaxed': 0.35,       # Alpha dominance for relaxation
            'beta_alert': 0.35,          # Beta dominance for alertness
            'theta_drowsy': 0.30         # Theta dominance for drowsiness
        }
    
    def compute_scientific_indices(self, band_powers: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute scientifically validated EEG indices
        
        Args:
            band_powers: Dictionary with keys 'delta', 'theta', 'alpha', 'beta', 'gamma'
        
        Returns:
            Dictionary of scientific indices
        """
        # Extract band powers
        delta = band_powers.get('delta', 0) + 1e-10  # Add epsilon to prevent division by zero
        theta = band_powers.get('theta', 0) + 1e-10
        alpha = band_powers.get('alpha', 0) + 1e-10
        beta = band_powers.get('beta', 0) + 1e-10
        gamma = band_powers.get('gamma', 0) + 1e-10
        
        total = delta + theta + alpha + beta + gamma
        
        # Normalize to relative powers (0-1 scale)
        delta_rel = delta / total
        theta_rel = theta / total
        alpha_rel = alpha / total
        beta_rel = beta / total
        gamma_rel = gamma / total
        
        # Compute scientific indices
        indices = {
            # Engagement Index (Pope et al., 1995)
            'engagement_index': beta / (alpha + theta),
            
            # Relaxation Index (alpha/beta ratio)
            'relaxation_index': alpha / beta,
            
            # Drowsiness Index
            'drowsiness_index': (theta + delta) / (alpha + beta),
            
            # Vigilance Index (alertness)
            'vigilance_index': (alpha + beta) / (theta + delta),
            
            # Mental Workload (theta/alpha ratio)
            'mental_workload': theta / alpha,
            
            # Cognitive Load (theta/beta ratio)
            'cognitive_load': theta / beta,
            
            # Focus Index (beta dominance with gamma)
            'focus_index': (beta + gamma) / (alpha + theta),
            
            # Relative powers for analysis
            'delta_relative': delta_rel,
            'theta_relative': theta_rel,
            'alpha_relative': alpha_rel,
            'beta_relative': beta_rel,
            'gamma_relative': gamma_rel
        }
        
        return indices
    
    def classify_brain_state(self, band_powers: Dict[str, float]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify brain state using multi-factor scientific analysis
        
        Args:
            band_powers: Dictionary with band powers (delta, theta, alpha, beta, gamma)
        
        Returns:
            Tuple of (state_name, confidence, indices_dict)
        """
        # Compute scientific indices
        indices = self.compute_scientific_indices(band_powers)
        
        # Extract relative powers
        delta_rel = indices['delta_relative']
        theta_rel = indices['theta_relative']
        alpha_rel = indices['alpha_relative']
        beta_rel = indices['beta_relative']
        gamma_rel = indices['gamma_relative']
        
        # Multi-factor classification with scientific rationale
        state = "Unknown State"
        confidence = 0.5
        rationale = []
        
        # Priority 1: Deep Sleep (Delta dominance > 40%)
        if delta_rel > self.thresholds['delta_sleep']:
            state = "Deep Sleep (Stage 3-4 NREM)"
            confidence = min(delta_rel / self.thresholds['delta_sleep'], 1.0)
            rationale = [f"Delta dominance: {delta_rel*100:.1f}%"]
        
        # Priority 2: Drowsiness/Light Sleep (High theta + drowsiness index)
        elif (theta_rel > self.thresholds['theta_drowsy'] and 
              indices['drowsiness_index'] > self.thresholds['drowsiness_high']):
            state = "Drowsy/Light Sleep (Stage 1-2)"
            confidence = min(indices['drowsiness_index'] / self.thresholds['drowsiness_high'], 1.0)
            rationale = [f"Theta: {theta_rel*100:.1f}%", f"Drowsiness index: {indices['drowsiness_index']:.2f}"]
        
        # Priority 3: Deep Relaxation (High alpha + high relaxation index)
        elif (alpha_rel > self.thresholds['alpha_relaxed'] and 
              indices['relaxation_index'] > self.thresholds['relaxation_high']):
            state = "Relaxed/Meditative (High Alpha)"
            confidence = min(indices['relaxation_index'] / self.thresholds['relaxation_high'], 1.0)
            rationale = [f"Alpha: {alpha_rel*100:.1f}%", f"Relaxation index: {indices['relaxation_index']:.2f}"]
        
        # Priority 4: High Focus/Concentration (High beta + gamma, high focus index)
        elif (beta_rel > self.thresholds['beta_alert'] and 
              indices['focus_index'] > self.thresholds['engagement_high']):
            state = "Highly Focused/Concentrated"
            confidence = min(indices['focus_index'] / self.thresholds['engagement_high'], 1.0)
            rationale = [f"Beta: {beta_rel*100:.1f}%", f"Focus index: {indices['focus_index']:.2f}"]
        
        # Priority 5: Active/Alert (High beta, high engagement)
        elif (beta_rel > 0.25 and 
              indices['engagement_index'] > self.thresholds['engagement_low']):
            state = "Active/Alert (High Beta)"
            confidence = min(indices['engagement_index'] / self.thresholds['engagement_high'], 1.0)
            rationale = [f"Beta: {beta_rel*100:.1f}%", f"Engagement: {indices['engagement_index']:.2f}"]
        
        # Priority 6: Calm/Wakeful (Moderate alpha)
        elif alpha_rel > 0.25:
            state = "Calm/Wakeful (Alpha Dominant)"
            confidence = alpha_rel / self.thresholds['alpha_relaxed']
            rationale = [f"Alpha: {alpha_rel*100:.1f}%"]
        
        # Priority 7: Mental Workload (High theta/alpha ratio)
        elif indices['mental_workload'] > 1.0:
            state = "Mental Workload/Processing"
            confidence = min(indices['mental_workload'] / 1.5, 1.0)
            rationale = [f"Mental workload index: {indices['mental_workload']:.2f}"]
        
        # Default: Transitional state
        else:
            state = "Transitional/Mixed State"
            # Find dominant band
            bands = {'Delta': delta_rel, 'Theta': theta_rel, 'Alpha': alpha_rel, 
                    'Beta': beta_rel, 'Gamma': gamma_rel}
            dominant = max(bands.items(), key=lambda x: x[1])
            confidence = dominant[1]
            rationale = [f"Dominant: {dominant[0]} ({dominant[1]*100:.1f}%)"]
        
        # Store rationale in indices for debugging
        indices['classification_rationale'] = "; ".join(rationale)
        indices['state'] = state
        indices['confidence'] = float(np.clip(confidence, 0.0, 1.0))
        
        return state, float(np.clip(confidence, 0.0, 1.0)), indices
    
    def get_state_description(self, state: str) -> str:
        """Get clinical description of brain state"""
        descriptions = {
            "Deep Sleep (Stage 3-4 NREM)": "Deep restorative sleep with delta wave dominance. Associated with physical recovery and memory consolidation.",
            "Drowsy/Light Sleep (Stage 1-2)": "Transitional state between wakefulness and sleep. Characterized by theta wave activity.",
            "Relaxed/Meditative (High Alpha)": "Calm, relaxed, wakeful state. Often seen during meditation or eyes-closed rest. Alpha waves indicate relaxed awareness.",
            "Highly Focused/Concentrated": "Intense concentration and cognitive processing. High beta and gamma activity indicates active problem-solving.",
            "Active/Alert (High Beta)": "Active thinking, alertness, and engagement. Beta waves indicate conscious cognitive activity.",
            "Calm/Wakeful (Alpha Dominant)": "Relaxed but alert state. Common during quiet wakefulness with eyes closed.",
            "Mental Workload/Processing": "Active cognitive processing with elevated theta. May indicate learning or working memory tasks.",
            "Transitional/Mixed State": "Mixed brain activity without clear dominant pattern. May indicate state transition or complex cognitive activity."
        }
        return descriptions.get(state, "Brain state classification based on EEG band power analysis.")
