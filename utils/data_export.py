"""Data Export Utilities with Role-Based Access Control"""
import json
import csv
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

class DataExporter:
    """Export EEG analysis data to various formats with role-based permissions"""
    
    def __init__(self, user_role: str = 'researcher'):
        self.user_role = user_role.lower()
    
    def check_permission(self, export_type: str) -> bool:
        """Check if user has permission for export type"""
        return PermissionManager.check_access(self.user_role, export_type)
    
    def export_to_csv(self, data: Dict, output_path: str, export_type: str = 'metrics') -> str:
        """
        Export data to CSV format
        
        Args:
            data: Dictionary with data to export
            output_path: Path for output file
            export_type: Type of export (raw_data, processed_data, metrics, predictions, full_report)
        
        Returns:
            Path to exported file
        """
        if not self.check_permission(export_type):
            raise PermissionError(f"Role '{self.user_role}' not permitted to export '{export_type}'")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert data to DataFrame based on type
        if export_type == 'metrics':
            df = self._metrics_to_dataframe(data)
        elif export_type == 'predictions':
            df = self._predictions_to_dataframe(data)
        elif export_type == 'processed_data':
            df = self._processed_data_to_dataframe(data)
        elif export_type == 'full_report':
            df = self._full_report_to_dataframe(data)
        else:
            df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def export_to_json(self, data: Dict, output_path: str, export_type: str = 'metrics') -> str:
        """
        Export data to JSON format
        
        Args:
            data: Dictionary with data to export
            output_path: Path for output file
            export_type: Type of export
        
        Returns:
            Path to exported file
        """
        if not self.check_permission(export_type):
            raise PermissionError(f"Role '{self.user_role}' not permitted to export '{export_type}'")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        export_data = {
            'metadata': {
                'export_type': export_type,
                'export_timestamp': datetime.now().isoformat(),
                'exported_by_role': self.user_role
            },
            'data': data
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return output_path
    
    def _metrics_to_dataframe(self, metrics: Dict) -> pd.DataFrame:
        """Convert metrics dictionary to DataFrame"""
        rows = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                rows.append({'metric': key, 'value': value})
        return pd.DataFrame(rows)
    
    def _predictions_to_dataframe(self, predictions: List[Dict]) -> pd.DataFrame:
        """Convert predictions list to DataFrame"""
        return pd.DataFrame(predictions)
    
    def _processed_data_to_dataframe(self, data: Dict) -> pd.DataFrame:
        """Convert processed EEG data to DataFrame"""
        if 'selected_data' in data:
            # Convert numpy array to DataFrame
            import numpy as np
            arr = np.array(data['selected_data'])
            if len(arr.shape) == 2:
                df = pd.DataFrame(arr.T)  # Transpose: samples x channels
                df.columns = [f"Channel_{i+1}" for i in range(arr.shape[0])]
                return df
        return pd.DataFrame(data)
    
    def _full_report_to_dataframe(self, report: Dict) -> pd.DataFrame:
        """Convert full report to DataFrame"""
        rows = []
        
        # Session info
        if 'session_info' in report:
            for key, value in report['session_info'].items():
                rows.append({'category': 'session', 'metric': key, 'value': value})
        
        # Metrics
        if 'metrics' in report:
            for key, value in report['metrics'].items():
                if isinstance(value, (int, float, str)):
                    rows.append({'category': 'metric', 'metric': key, 'value': value})
        
        # Predictions
        if 'predictions' in report:
            for pred in report['predictions']:
                model_name = pred.get('model_name', 'unknown')
                for key, value in pred.items():
                    if key != 'model_name':
                        rows.append({'category': 'prediction', 'metric': f"{model_name}_{key}", 'value': value})
        
        return pd.DataFrame(rows)
    
    def batch_export(self, data_list: List[Dict], output_dir: str, 
                    export_format: str = 'csv', export_type: str = 'metrics') -> List[str]:
        """
        Export multiple datasets
        
        Args:
            data_list: List of data dictionaries
            output_dir: Output directory
            export_format: 'csv' or 'json'
            export_type: Type of export
        
        Returns:
            List of exported file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        for i, data in enumerate(data_list):
            file_name = f"export_{i+1}.{export_format}"
            output_path = str(Path(output_dir) / file_name)
            
            if export_format == 'csv':
                path = self.export_to_csv(data, output_path, export_type)
            elif export_format == 'json':
                path = self.export_to_json(data, output_path, export_type)
            else:
                raise ValueError(f"Unsupported format: {export_format}")
            
            exported_files.append(path)
        
        return exported_files
    
    def get_export_summary(self, exported_files: List[str]) -> Dict:
        """Generate summary of exported files"""
        total_size = sum([Path(f).stat().st_size for f in exported_files if Path(f).exists()])
        
        return {
            'total_files': len(exported_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'files': exported_files
        }


class PermissionManager:
    """Manage role-based permissions for data access"""
    
    ROLE_HIERARCHY = {
        'admin': 3,
        'doctor': 2,
        'researcher': 1
    }
    
    DATA_ACCESS_LEVELS = {
        'raw_data': 3,        # Admin only
        'processed_data': 2,  # Doctor and above
        'metrics': 1,         # All roles
        'predictions': 1,     # All roles
        'full_report': 2,     # Doctor and above
        'user_management': 3  # Admin only
    }
    
    @classmethod
    def check_access(cls, user_role: str, data_type: str) -> bool:
        """Check if role has access to data type"""
        role_level = cls.ROLE_HIERARCHY.get(user_role.lower(), 0)
        required_level = cls.DATA_ACCESS_LEVELS.get(data_type, 99)
        return role_level >= required_level
    
    @classmethod
    def get_accessible_data_types(cls, user_role: str) -> List[str]:
        """Get list of data types accessible to role"""
        role_level = cls.ROLE_HIERARCHY.get(user_role.lower(), 0)
        accessible = []
        
        for data_type, required_level in cls.DATA_ACCESS_LEVELS.items():
            if role_level >= required_level:
                accessible.append(data_type)
        
        return accessible
