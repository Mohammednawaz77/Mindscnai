import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from typing import Tuple, Dict

class ClassicalMLModels:
    """Classical Machine Learning models for comparison"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None  # Will be initialized adaptively
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[SVC, float, Dict]:
        """Train classical Support Vector Machine"""
        start_time = time.time()
        
        # Preprocess
        X_scaled = self.scaler.fit_transform(X_train)
        # Initialize PCA adaptively based on data size
        n_components = min(10, X_scaled.shape[0], X_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Train SVM
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(X_reduced, y_train)
        
        # Training accuracy
        train_predictions = svm.predict(X_reduced)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        processing_time = time.time() - start_time
        
        metrics = {
            'training_accuracy': train_accuracy,
            'processing_time': processing_time,
            'n_support_vectors': len(svm.support_),
            'kernel': 'rbf'
        }
        
        return svm, train_accuracy, metrics
    
    def predict_svm(self, svm: SVC, X_test: np.ndarray, y_test: np.ndarray | None = None) -> Tuple[np.ndarray, float, Dict]:
        """Predict using SVM"""
        start_time = time.time()
        
        # Preprocess
        X_scaled = self.scaler.transform(X_test)
        X_reduced = self.pca.transform(X_scaled)
        
        # Predict
        predictions = svm.predict(X_reduced)
        
        processing_time = time.time() - start_time
        
        metrics = {'processing_time': processing_time}
        
        if y_test is not None:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        return predictions, processing_time, metrics
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           n_estimators: int = 100) -> Tuple[RandomForestClassifier, float, Dict]:
        """Train Random Forest classifier"""
        start_time = time.time()
        
        # Preprocess
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=42)
        rf.fit(X_scaled, y_train)
        
        # Training accuracy
        train_predictions = rf.predict(X_scaled)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        processing_time = time.time() - start_time
        
        metrics = {
            'training_accuracy': train_accuracy,
            'processing_time': processing_time,
            'n_estimators': n_estimators,
            'feature_importances': rf.feature_importances_.tolist()
        }
        
        return rf, train_accuracy, metrics
    
    def predict_random_forest(self, rf: RandomForestClassifier, X_test: np.ndarray, 
                             y_test: np.ndarray | None = None) -> Tuple[np.ndarray, float, Dict]:
        """Predict using Random Forest"""
        start_time = time.time()
        
        # Preprocess
        X_scaled = self.scaler.transform(X_test)
        
        # Predict
        predictions = rf.predict(X_scaled)
        
        processing_time = time.time() - start_time
        
        metrics = {'processing_time': processing_time}
        
        if y_test is not None:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        return predictions, processing_time, metrics
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract statistical features from EEG signals"""
        features = []
        
        for signal in X:
            # Time domain features
            mean = np.mean(signal)
            std = np.std(signal)
            var = np.var(signal)
            min_val = np.min(signal)
            max_val = np.max(signal)
            range_val = max_val - min_val
            
            # Frequency domain features (using FFT)
            fft = np.fft.fft(signal)
            fft_magnitude = np.abs(fft[:len(fft)//2])
            
            # Spectral features
            spectral_mean = np.mean(fft_magnitude)
            spectral_std = np.std(fft_magnitude)
            spectral_max = np.max(fft_magnitude)
            
            feature_vector = [
                mean, std, var, min_val, max_val, range_val,
                spectral_mean, spectral_std, spectral_max
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
