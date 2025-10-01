import hashlib
import secrets
from cryptography.fernet import Fernet
import os
import pickle
import numpy as np

class SecurityManager:
    """Handle data encryption and security"""
    
    def __init__(self):
        # Generate or load encryption key
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self) -> bytes:
        """Get existing key or create new one"""
        key_file = 'encryption.key'
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt_data(self, data: np.ndarray) -> bytes:
        """Encrypt numpy array data"""
        # Serialize data
        data_bytes = pickle.dumps(data)
        
        # Encrypt
        encrypted = self.cipher.encrypt(data_bytes)
        
        return encrypted
    
    def decrypt_data(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt data back to numpy array"""
        # Decrypt
        decrypted_bytes = self.cipher.decrypt(encrypted_data)
        
        # Deserialize
        data = pickle.loads(decrypted_bytes)
        
        return data
    
    def hash_data(self, data: np.ndarray) -> str:
        """Create hash of data for integrity verification"""
        data_bytes = pickle.dumps(data)
        hash_object = hashlib.sha256(data_bytes)
        return hash_object.hexdigest()
    
    def verify_data_integrity(self, data: np.ndarray, expected_hash: str) -> bool:
        """Verify data integrity using hash"""
        actual_hash = self.hash_data(data)
        return actual_hash == expected_hash
    
    def generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
