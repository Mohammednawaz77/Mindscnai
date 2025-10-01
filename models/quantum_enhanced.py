"""Enhanced Quantum ML Models with Configurable Architecture"""
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from typing import Tuple, Dict
import time

class EnhancedQuantumML:
    """Enhanced Quantum ML with deeper circuits and configurable architecture"""
    
    def __init__(self, n_qubits: int = 4, circuit_depth: int = 3, entanglement: str = 'circular'):
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.entanglement = entanglement
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.scaler = StandardScaler()
        self.pca: PCA = None  # type: ignore
    
    def hardware_efficient_ansatz(self, weights, layer_idx: int):
        """Hardware-efficient ansatz with configurable entanglement"""
        # Single-qubit rotations
        for i in range(self.n_qubits):
            qml.RY(weights[layer_idx, i, 0], wires=i)
            qml.RZ(weights[layer_idx, i, 1], wires=i)
        
        # Entangling gates
        if self.entanglement == 'circular':
            # Full circular entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
        elif self.entanglement == 'linear':
            # Linear entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        elif self.entanglement == 'full':
            # All-to-all entanglement
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
    
    def enhanced_feature_map(self, x):
        """Enhanced feature encoding with angle embedding and entanglement"""
        # Angle embedding
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
            qml.RZ(x[i] ** 2, wires=i)  # Non-linear encoding
        
        # Feature entanglement
        for i in range(self.n_qubits - 1):
            qml.CRZ(x[i] * x[i + 1], wires=[i, i + 1])
    
    def deep_quantum_kernel(self, x1, x2):
        """Deep quantum kernel with enhanced feature map"""
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            self.enhanced_feature_map(x1)
            qml.adjoint(self.enhanced_feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))
        
        probs = kernel_circuit(x1, x2)
        return probs[0]
    
    def quantum_kernel_matrix(self, X1, X2):
        """Compute quantum kernel matrix with enhanced kernel"""
        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                kernel_matrix[i, j] = self.deep_quantum_kernel(X1[i], X2[j])
        
        return kernel_matrix
    
    def train_enhanced_qsvm(self, X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0) -> Tuple[SVC, float, Dict]:
        """Train enhanced QSVM with deep quantum kernel"""
        start_time = time.time()
        
        # Adaptive PCA
        X_scaled = self.scaler.fit_transform(X_train)
        n_components = min(self.n_qubits, X_scaled.shape[0], X_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Compute enhanced quantum kernel
        K_train = self.quantum_kernel_matrix(X_reduced, X_reduced)
        
        # Train SVM
        qsvm = SVC(kernel='precomputed', C=C)
        qsvm.fit(K_train, y_train)
        
        train_score = qsvm.score(K_train, y_train)
        processing_time = time.time() - start_time
        
        metrics = {
            'training_accuracy': train_score,
            'processing_time': processing_time,
            'n_support_vectors': len(qsvm.support_),
            'n_qubits': self.n_qubits,
            'circuit_depth': self.circuit_depth,
            'entanglement': self.entanglement,
            'model_type': 'enhanced_qsvm'
        }
        
        return qsvm, train_score, metrics
    
    def predict_enhanced_qsvm(self, qsvm: SVC, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict using enhanced QSVM"""
        start_time = time.time()
        
        X_train_scaled = self.scaler.transform(X_train)
        X_train_reduced = self.pca.transform(X_train_scaled)
        
        X_test_scaled = self.scaler.transform(X_test)
        X_test_reduced = self.pca.transform(X_test_scaled)
        
        K_test = self.quantum_kernel_matrix(X_test_reduced, X_train_reduced)
        predictions = qsvm.predict(K_test)
        
        processing_time = time.time() - start_time
        
        return predictions, processing_time
    
    def get_circuit_info(self) -> Dict:
        """Get circuit configuration information"""
        return {
            'n_qubits': self.n_qubits,
            'circuit_depth': self.circuit_depth,
            'entanglement_type': self.entanglement,
            'total_parameters': self.circuit_depth * self.n_qubits * 2,
            'gate_count_estimate': self.circuit_depth * self.n_qubits * 3  # Approx gates per layer
        }
