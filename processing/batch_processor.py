"""Batch Processing for Multiple EEG Files"""
import concurrent.futures
import threading
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

from preprocessing.eeg_processor import EEGProcessor
from preprocessing.channel_selector import ChannelSelector
from preprocessing.advanced_signal_processing import AdvancedSignalProcessor
from models.quantum_ml import QuantumMLModels
from models.classical_ml import ClassicalMLModels
from analysis.brain_metrics import BrainMetricsAnalyzer
from reports.pdf_generator import PDFReportGenerator
from utils.helpers import Helpers

# Global lock for thread-safe EDF file reading
_edf_read_lock = threading.Lock()


class BatchEEGProcessor:
    """Batch processing for multiple EEG files with concurrent analysis"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.eeg_processor = EEGProcessor()
        self.channel_selector = ChannelSelector(target_channels=20)
        
    def process_single_file(self, file_path: str, 
                           apply_advanced_processing: bool = True,
                           generate_pdf: bool = True) -> Dict:
        """
        Process a single EEG file
        
        Args:
            file_path: Path to EDF file
            apply_advanced_processing: Apply advanced signal processing
            generate_pdf: Generate PDF report
        
        Returns:
            Dictionary with processing results
        """
        try:
            start_time = time.time()
            file_name = Path(file_path).name
            
            # Step 1: Load and preprocess (thread-safe EDF reading)
            with _edf_read_lock:
                data, channel_names, sampling_rate = self.eeg_processor.read_edf_file(file_path)
            preprocessed_data = self.eeg_processor.preprocess_signals(data)
            
            # Step 2: Channel selection
            selected_data, selected_channels, method_used = self.channel_selector.select_optimal_channels(
                preprocessed_data, channel_names, method='names'
            )
            
            # Step 3: Advanced signal processing (optional)
            if apply_advanced_processing:
                adv_processor = AdvancedSignalProcessor(sampling_rate=sampling_rate)
                selected_data, proc_info = adv_processor.apply_advanced_preprocessing(
                    selected_data,
                    remove_artifacts=True,
                    adaptive_filter=True,
                    detect_bad=True
                )
            else:
                proc_info = {}
            
            # Step 4: Extract features
            classical_ml = ClassicalMLModels()
            features = classical_ml.extract_features(selected_data)
            
            # Step 5: Brain metrics
            brain_analyzer = BrainMetricsAnalyzer(sampling_rate=sampling_rate)
            metrics = brain_analyzer.compute_multi_channel_metrics(selected_data)
            # Extract average metrics for reporting
            metrics = metrics.get('average', {})
            
            # Step 6: ML predictions (using dummy labels for demo)
            labels = Helpers.create_dummy_labels(len(features))
            X_train, X_test, y_train, y_test = Helpers.split_data(features, labels, test_size=0.3)
            
            predictions = []
            
            # Classical models
            svm_model, svm_acc = classical_ml.train_svm(X_train, y_train)
            svm_pred, svm_time = classical_ml.predict_svm(svm_model, X_test)
            predictions.append({
                'model_name': 'SVM',
                'model_type': 'classical',
                'accuracy': svm_acc,
                'processing_time': svm_time
            })
            
            # Quantum model (QSVM)
            try:
                qml = QuantumMLModels(n_qubits=4)
                qsvm, qsvm_acc, qsvm_metrics = qml.train_qsvm(X_train, y_train)
                qsvm_pred, qsvm_time = qml.predict_qsvm(qsvm, X_train, X_test)
                predictions.append({
                    'model_name': 'QSVM',
                    'model_type': 'quantum',
                    'accuracy': qsvm_acc,
                    'processing_time': qsvm_time
                })
            except Exception as e:
                print(f"QSVM failed for {file_name}: {str(e)[:50]}")
            
            # Step 7: Generate PDF (optional)
            pdf_path = None
            if generate_pdf:
                try:
                    session_data = {
                        'filename': file_name,
                        'upload_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'channels_original': data.shape[0],
                        'channels_selected': selected_data.shape[0],
                        'processing_status': 'completed'
                    }
                    
                    pdf_gen = PDFReportGenerator()
                    Helpers.ensure_directory('reports_output/batch')
                    pdf_path = f"reports_output/batch/{Path(file_name).stem}_report.pdf"
                    pdf_gen.create_report(session_data, metrics, predictions, pdf_path)
                except Exception as e:
                    print(f"PDF generation failed for {file_name}: {str(e)[:50]}")
            
            processing_time = time.time() - start_time
            
            return {
                'file_name': file_name,
                'status': 'success',
                'processing_time': processing_time,
                'channels_original': data.shape[0],
                'channels_selected': selected_data.shape[0],
                'brain_metrics': metrics,
                'predictions': predictions,
                'advanced_processing': proc_info,
                'pdf_path': pdf_path
            }
            
        except Exception as e:
            return {
                'file_name': Path(file_path).name,
                'status': 'failed',
                'error': str(e),
                'processing_time': 0
            }
    
    def process_batch_sequential(self, file_paths: List[str], **kwargs) -> List[Dict]:
        """Process files sequentially"""
        results = []
        for file_path in file_paths:
            result = self.process_single_file(file_path, **kwargs)
            results.append(result)
        return results
    
    def process_batch_parallel(self, file_paths: List[str], **kwargs) -> List[Dict]:
        """Process files in parallel using ThreadPoolExecutor"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_path, **kwargs): file_path 
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'file_name': Path(file_path).name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        return results
    
    def get_batch_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics for batch processing"""
        total_files = len(results)
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        total_time = sum([r.get('processing_time', 0) for r in results])
        avg_time = total_time / total_files if total_files > 0 else 0
        
        # Average accuracies
        all_accuracies = []
        for result in successful:
            for pred in result.get('predictions', []):
                all_accuracies.append(pred.get('accuracy', 0))
        
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0
        
        summary = {
            'total_files': total_files,
            'successful': len(successful),
            'failed': len(failed),
            'total_processing_time': total_time,
            'average_processing_time': avg_time,
            'average_accuracy': avg_accuracy,
            'success_rate': len(successful) / total_files if total_files > 0 else 0
        }
        
        return summary
