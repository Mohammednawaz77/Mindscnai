"""Quantum Machine Learning Configuration"""

class QuantumConfig:
    """Configuration for Quantum ML models"""
    
    # QSVM Configuration
    QSVM_N_QUBITS = 4
    QSVM_C_PARAMETER = 1.0
    
    # VQC Configuration  
    VQC_N_QUBITS = 4
    VQC_N_LAYERS = 3  # Increased depth for better expressiveness
    VQC_EPOCHS = 50
    VQC_LEARNING_RATE = 0.1
    
    # Feature Processing
    PCA_VARIANCE_THRESHOLD = 0.95  # Keep 95% of variance
    
    # Circuit Architecture
    ENTANGLEMENT_TYPE = 'circular'  # 'linear' or 'circular'
    USE_HARDWARE_EFFICIENT = True
    
    @classmethod
    def get_qsvm_config(cls):
        """Get QSVM configuration"""
        return {
            'n_qubits': cls.QSVM_N_QUBITS,
            'C': cls.QSVM_C_PARAMETER
        }
    
    @classmethod
    def get_vqc_config(cls):
        """Get VQC configuration"""
        return {
            'n_qubits': cls.VQC_N_QUBITS,
            'n_layers': cls.VQC_N_LAYERS,
            'epochs': cls.VQC_EPOCHS,
            'learning_rate': cls.VQC_LEARNING_RATE
        }
