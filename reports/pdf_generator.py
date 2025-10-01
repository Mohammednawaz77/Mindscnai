from fpdf import FPDF
from datetime import datetime
from typing import Dict, List
import numpy as np
import os

class PDFReportGenerator:
    """Generate PDF reports for BCI analysis sessions"""
    
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
    
    def create_report(self, session_data: Dict, metrics: Dict, 
                     predictions: List[Dict], output_path: str) -> str:
        """Create comprehensive PDF report"""
        
        # Add page
        self.pdf.add_page()
        
        # Title
        self._add_title()
        
        # Session information
        self._add_session_info(session_data)
        
        # Brain metrics
        self._add_brain_metrics(metrics)
        
        # Model predictions
        self._add_predictions(predictions)
        
        # Summary
        self._add_summary(metrics, predictions)
        
        # Save PDF
        self.pdf.output(output_path)
        
        return output_path
    
    def _add_title(self):
        """Add report title"""
        self.pdf.set_font('Arial', 'B', 24)
        self.pdf.cell(0, 20, 'QuantumBCI Analysis Report', 0, 1, 'C')
        
        self.pdf.set_font('Arial', '', 12)
        self.pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.pdf.ln(10)
    
    def _add_session_info(self, session_data: Dict):
        """Add session information"""
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Session Information', 0, 1)
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', '', 12)
        
        info_items = [
            ('Filename', session_data.get('filename', 'N/A')),
            ('Upload Time', session_data.get('upload_time', 'N/A')),
            ('Original Channels', str(session_data.get('channels_original', 'N/A'))),
            ('Selected Channels', str(session_data.get('channels_selected', 'N/A'))),
            ('Processing Status', session_data.get('processing_status', 'N/A'))
        ]
        
        for label, value in info_items:
            self.pdf.cell(60, 8, f'{label}:', 0, 0)
            self.pdf.cell(0, 8, str(value), 0, 1)
        
        self.pdf.ln(10)
    
    def _add_brain_metrics(self, metrics: Dict):
        """Add brain metrics section"""
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Brain Metrics Analysis', 0, 1)
        self.pdf.ln(5)
        
        # Brain state
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(60, 8, 'Brain State:', 0, 0)
        self.pdf.set_font('Arial', '', 12)
        self.pdf.cell(0, 8, metrics.get('brain_state', 'Unknown'), 0, 1)
        self.pdf.ln(5)
        
        # Band powers table
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.cell(0, 10, 'Frequency Band Powers', 0, 1)
        
        self.pdf.set_font('Arial', 'B', 11)
        self.pdf.cell(60, 8, 'Band', 1, 0, 'C')
        self.pdf.cell(60, 8, 'Absolute Power (uV^2)', 1, 0, 'C')
        self.pdf.cell(60, 8, 'Relative Power (%)', 1, 1, 'C')
        
        self.pdf.set_font('Arial', '', 11)
        
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        for band in bands:
            self.pdf.cell(60, 8, band.capitalize(), 1, 0)
            self.pdf.cell(60, 8, f"{metrics.get(f'{band}_power', 0):.4f}", 1, 0, 'C')
            self.pdf.cell(60, 8, f"{metrics.get(f'{band}_relative', 0):.2f}%", 1, 1, 'C')
        
        self.pdf.ln(10)
    
    def _add_predictions(self, predictions: List[Dict]):
        """Add model predictions section"""
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Model Predictions', 0, 1)
        self.pdf.ln(5)
        
        # Predictions table with prediction result
        self.pdf.set_font('Arial', 'B', 11)
        self.pdf.cell(45, 8, 'Model', 1, 0, 'C')
        self.pdf.cell(60, 8, 'Prediction', 1, 0, 'C')
        self.pdf.cell(40, 8, 'Accuracy', 1, 0, 'C')
        self.pdf.cell(45, 8, 'Time (s)', 1, 1, 'C')
        
        self.pdf.set_font('Arial', '', 11)
        
        for pred in predictions:
            self.pdf.cell(45, 8, pred.get('model_name', 'N/A'), 1, 0)
            
            # Show prediction result
            pred_result = pred.get('prediction_result', 'Unknown')
            self.pdf.cell(60, 8, str(pred_result), 1, 0, 'C')
            
            accuracy = pred.get('accuracy', 0)
            self.pdf.cell(40, 8, f"{accuracy:.2%}" if accuracy else 'N/A', 1, 0, 'C')
            
            proc_time = pred.get('processing_time', 0)
            self.pdf.cell(45, 8, f"{proc_time:.3f}" if proc_time else 'N/A', 1, 1, 'C')
        
        self.pdf.ln(10)
    
    def _add_summary(self, metrics: Dict, predictions: List[Dict]):
        """Add summary section"""
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Summary & Recommendations', 0, 1)
        self.pdf.ln(5)
        
        self.pdf.set_font('Arial', '', 12)
        
        # Best performing model
        if predictions:
            best_model = max(predictions, key=lambda x: x.get('accuracy', 0))
            self.pdf.multi_cell(0, 8, f"- Best performing model: {best_model.get('model_name', 'N/A')} "
                                     f"with {best_model.get('accuracy', 0):.2%} accuracy")
        
        # Brain state interpretation
        brain_state = metrics.get('brain_state', 'Unknown')
        self.pdf.multi_cell(0, 8, f"- Detected brain state: {brain_state}")
        
        # Dominant frequency band
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_powers = {band: metrics.get(f'{band}_power', 0) for band in bands}
        dominant_band = max(band_powers.items(), key=lambda x: x[1])[0]
        self.pdf.multi_cell(0, 8, f"- Dominant frequency band: {dominant_band.capitalize()}")
        
        # Total power
        total_power = metrics.get('total_power', 0)
        self.pdf.multi_cell(0, 8, f"- Total spectral power: {total_power:.4f} uV^2")
        
        self.pdf.ln(5)
        
        # Recommendations
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.cell(0, 10, 'Clinical Recommendations:', 0, 1)
        
        self.pdf.set_font('Arial', '', 12)
        
        recommendations = self._generate_recommendations(metrics)
        for rec in recommendations:
            self.pdf.multi_cell(0, 8, f"- {rec}")
        
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate clinical recommendations based on metrics"""
        recommendations = []
        
        # Alpha power recommendations
        alpha_rel = metrics.get('alpha_relative', 0)
        if alpha_rel > 40:
            recommendations.append("High alpha activity detected - suggests relaxed, wakeful state")
        elif alpha_rel < 20:
            recommendations.append("Low alpha activity - may indicate stress or active cognitive processing")
        
        # Beta power recommendations
        beta_rel = metrics.get('beta_relative', 0)
        if beta_rel > 40:
            recommendations.append("Elevated beta activity - suggests high alertness or anxiety")
        
        # Theta power recommendations
        theta_rel = metrics.get('theta_relative', 0)
        if theta_rel > 30:
            recommendations.append("Increased theta activity - may indicate drowsiness or deep meditation")
        
        # General recommendation
        recommendations.append("Continue monitoring for trend analysis")
        recommendations.append("Consult with neurologist for clinical interpretation")
        
        return recommendations
