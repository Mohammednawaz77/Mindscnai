import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
import json

class DatabaseManager:
    """Manages database operations for the BCI application"""
    
    def __init__(self, db_path: str = "quantumbci.db"):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def initialize_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                channels_original INTEGER,
                channels_selected INTEGER,
                processing_status TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                accuracy REAL,
                prediction_result TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # Brain metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS brain_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                alpha_power REAL,
                beta_power REAL,
                theta_power REAL,
                delta_power REAL,
                total_power REAL,
                metrics_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # Activity logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # EEG data table (encrypted storage)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS eeg_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                data_encrypted BLOB,
                data_hash TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, password_hash: str, full_name: str, 
                   email: str, role: str) -> Optional[int]:
        """Create a new user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, password_hash, full_name, email, role)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, password_hash, full_name, email, role))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def update_last_login(self, user_id: int):
        """Update user's last login time"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        ''', (user_id,))
        conn.commit()
        conn.close()
    
    def create_session(self, user_id: int, filename: str, channels_original: int, 
                      channels_selected: int) -> int:
        """Create a new analysis session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (user_id, filename, channels_original, 
                                channels_selected, processing_status)
            VALUES (?, ?, ?, ?, 'processing')
        ''', (user_id, filename, channels_original, channels_selected))
        conn.commit()
        if not cursor.lastrowid:
            conn.close()
            raise Exception("Failed to create session - no ID returned")
        session_id = cursor.lastrowid
        conn.close()
        return session_id
    
    def update_session_status(self, session_id: int, status: str):
        """Update session processing status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sessions SET processing_status = ? WHERE id = ?
        ''', (status, session_id))
        conn.commit()
        conn.close()
    
    def save_prediction(self, session_id: int, model_type: str, model_name: str,
                       accuracy: float, prediction_result: str, processing_time: float) -> int:
        """Save prediction results"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (session_id, model_type, model_name, accuracy,
                                   prediction_result, processing_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, model_type, model_name, accuracy, prediction_result, processing_time))
        conn.commit()
        if not cursor.lastrowid:
            conn.close()
            raise Exception("Failed to save prediction - no ID returned")
        pred_id = cursor.lastrowid
        conn.close()
        return pred_id
    
    def save_brain_metrics(self, session_id: int, alpha: float, beta: float,
                          theta: float, delta: float, total: float, metrics_dict: Dict):
        """Save brain metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO brain_metrics (session_id, alpha_power, beta_power,
                                      theta_power, delta_power, total_power, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, alpha, beta, theta, delta, total, json.dumps(metrics_dict)))
        conn.commit()
        conn.close()
    
    def save_eeg_data(self, session_id: int, data_encrypted: bytes, data_hash: str):
        """Save encrypted EEG data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO eeg_data (session_id, data_encrypted, data_hash)
            VALUES (?, ?, ?)
        ''', (session_id, data_encrypted, data_hash))
        conn.commit()
        conn.close()
    
    def log_activity(self, user_id: int, action: str, details: str):
        """Log user activity"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO activity_logs (user_id, action, details)
            VALUES (?, ?, ?)
        ''', (user_id, action, details))
        conn.commit()
        conn.close()
    
    def get_user_sessions(self, user_id: int) -> List[Dict]:
        """Get all sessions for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM sessions WHERE user_id = ? ORDER BY upload_time DESC
        ''', (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_session_predictions(self, session_id: int) -> List[Dict]:
        """Get predictions for a session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM predictions WHERE session_id = ? ORDER BY created_at
        ''', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_session_metrics(self, session_id: int) -> Optional[Dict]:
        """Get brain metrics for a session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM brain_metrics WHERE session_id = ?
        ''', (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_recent_activity(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get recent activity logs for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM activity_logs WHERE user_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        ''', (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_all_users(self) -> List[Dict]:
        """Get all users (admin only)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, full_name, email, role, created_at, last_login FROM users')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_user_session_count(self, user_id: int) -> int:
        """Get session count for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) as count FROM sessions WHERE user_id = ?', (user_id,))
        count = cursor.fetchone()['count']
        conn.close()
        return count
    
    def get_all_activity_logs(self, limit: int = 50) -> List[Dict]:
        """Get all activity logs (admin only)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT al.*, u.username, u.full_name 
            FROM activity_logs al
            JOIN users u ON al.user_id = u.id
            ORDER BY al.timestamp DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
