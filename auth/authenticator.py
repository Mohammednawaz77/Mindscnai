import bcrypt
from typing import Optional, Dict, Tuple
from database.db_manager import DatabaseManager

class AuthManager:
    """Handles user authentication and authorization"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def register_user(self, username: str, password: str, full_name: str,
                     email: str, role: str = 'researcher') -> Tuple[bool, str]:
        """Register a new user"""
        
        # Validate role
        valid_roles = ['admin', 'doctor', 'researcher']
        if role.lower() not in valid_roles:
            return False, "Invalid role"
        
        # Hash password
        password_hash = self.hash_password(password)
        
        # Create user
        user_id = self.db_manager.create_user(
            username, password_hash, full_name, email, role.lower()
        )
        
        if user_id:
            return True, "User registered successfully"
        else:
            return False, "Username or email already exists"
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate a user"""
        user = self.db_manager.get_user_by_username(username)
        
        if user and self.verify_password(password, user['password_hash']):
            # Update last login
            self.db_manager.update_last_login(user['id'])
            
            # Remove password hash from returned user
            user_data = {k: v for k, v in user.items() if k != 'password_hash'}
            return user_data
        
        return None
    
    def check_role(self, user: Dict, required_role: str) -> bool:
        """Check if user has required role"""
        role_hierarchy = {
            'admin': 3,
            'doctor': 2,
            'researcher': 1
        }
        
        user_level = role_hierarchy.get(user['role'], 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    def is_admin(self, user: Dict) -> bool:
        """Check if user is admin"""
        return user['role'] == 'admin'
