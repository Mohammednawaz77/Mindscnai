from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class User:
    """User model"""
    id: int
    username: str
    password_hash: str
    full_name: str
    email: str
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None

@dataclass
class Session:
    """Analysis session model"""
    id: int
    user_id: int
    filename: str
    upload_time: datetime
    channels_original: int
    channels_selected: int
    processing_status: str

@dataclass
class Prediction:
    """Prediction result model"""
    id: int
    session_id: int
    model_type: str
    model_name: str
    accuracy: float
    prediction_result: str
    processing_time: float
    created_at: datetime

@dataclass
class BrainMetrics:
    """Brain metrics model"""
    id: int
    session_id: int
    alpha_power: float
    beta_power: float
    theta_power: float
    delta_power: float
    total_power: float
    metrics_json: str

@dataclass
class ActivityLog:
    """Activity log model"""
    id: int
    user_id: int
    action: str
    details: str
    timestamp: datetime
