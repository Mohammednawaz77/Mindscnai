# QuantumBCI - Signal Prediction & Real-Time Monitoring System

## Overview

QuantumBCI is a brain-computer interface (BCI) analysis platform that leverages quantum machine learning to predict and analyze EEG signals. The system processes 64-channel EEG data from EDF files, applies advanced signal processing techniques, and uses both quantum and classical machine learning models for comparison and prediction.

The application features **real-time continuous EEG streaming** with live visualization of brain states including cognitive load, focus, and anxiety detection using machine learning techniques. Blue line charts update continuously to show 20-channel signal flows with sliding-window inference.

The application is built as a multi-page Streamlit web application with role-based access control, supporting researchers, doctors, and administrators with different permission levels for data access and analysis capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent UI/UX Improvements (Oct 2025)

**Latest Updates (Oct 2, 2025):**
1. **WebSocket-Based Flicker-Free Streaming** (`websocket_server.py`, `components/websocket_signal_viewer.py`): 
   - Dual-server architecture: Streamlit (port 5000) + FastAPI WebSocket (port 8000)
   - Zero flickering with dedicated WebSocket connection for signal visualization
   - ALL 20 channels displayed simultaneously in hospital ECG style
   - Dark background (#0a0a0a) with cyan signals (#00D9FF) like medical monitors
   - Continuous 5-second streaming window at 5Hz broadcast rate
   - Signal display at BOTTOM of page for better workflow
   - Vertical channel stacking with labels on left side
   - Grid lines every 0.5 seconds for time reference
   - WebSocket bridge module for thread-safe ring buffer sharing
   - All brain metrics, classification, and functionality unchanged

2. **Scientific Brain State Classification**:
   - Implemented validated neuroscience indices (Engagement, Relaxation, Drowsiness, Vigilance)
   - Per-channel analysis with robust median aggregation
   - 8 distinct brain states: Deep Sleep, Drowsy, Relaxed, Focused, Active, Calm, Mental Workload, Transitional
   - Brain state saved in BOTH predictions and brain_metrics tables
   - Fixed "N/A" and "Unknown" issues in Results page and PDF reports

2. **Enhanced Login Security & UX**:
   - Hidden sidebar navigation before login (CSS: `[data-testid="stSidebar"]`)
   - Improved login text visibility: white text with heavy shadows (3px 3px 6px) against neural network background
   - Removed insecure Remember Me feature (was URL-based, security vulnerability)
   - Sidebar auto-expands after successful authentication

3. **Archived Pages**:
   - Old Upload/Analysis page → `2_Upload_Analysis.py.old`
   - Old Live Stream page → `4_Live_Stream.py.old`
   - Navigation simplified to single Predict Model entry point

**Previous Enhancements:**
1. **Login Page**: Neural network background image with professional styling and dark overlay
2. **Simplified Upload Flow**: Removed verbose displays, unified workflow, clear progress indicators
3. **Fixed Calibration**: Shows exact time remaining (e.g., "Need 14.7s more")
4. **Performance Optimization**: Streamlined processing with visual feedback

## System Architecture

### Frontend Architecture

**Problem**: Need for an accessible, interactive interface for EEG analysis with real-time visualization
**Solution**: Multi-page Streamlit application with role-based dashboards and real-time streaming
**Rationale**: Streamlit provides rapid prototyping with built-in state management and Python-native visualization libraries

- **Main Entry Point** (`app.py`): Handles authentication flow with hidden sidebar pre-login, routes to main application
- **Page Structure**: Dashboard, **Predict Model (unified upload/stream)**, Model Comparison, Results, User Management, Activity Logs
- **Visualization**: Plotly for interactive charts with continuous updates, blue line charts for real-time signals, gauge displays for brain states
- **State Management**: Streamlit session state for user authentication, database connections, analysis results, and **streaming buffers**
- **Real-Time Updates**: Auto-refresh at 5Hz for smooth live visualization of continuous EEG signals
- **Security**: Sidebar hidden on login page, no persistent session tokens (session-based auth only)

### Authentication & Authorization

**Problem**: Secure multi-user access with role-based permissions
**Solution**: Custom authentication system with bcrypt password hashing and SQLite-based user management
**Alternatives Considered**: OAuth/third-party auth (rejected for simpler deployment)

- **Password Security**: bcrypt hashing with salt for secure password storage
- **Role-Based Access Control**: Three user roles (admin, doctor, researcher) with different data export and management permissions
- **Session Management**: Login state persisted in Streamlit session state with last login tracking

### Data Processing Pipeline

**Problem**: Process high-dimensional 64-channel EEG data efficiently
**Solution**: Multi-stage preprocessing pipeline with channel selection and artifact removal

1. **EEG File Loading** (`preprocessing/eeg_processor.py`): PyEDFlib for EDF file parsing
2. **Signal Preprocessing**: Bandpass filtering (0.5-50 Hz), notch filtering (50 Hz power line noise), normalization
3. **Channel Selection** (`preprocessing/channel_selector.py`): Reduces 64 channels to 20 optimal channels based on 10-20 system priority
4. **Advanced Signal Processing** (`preprocessing/advanced_signal_processing.py`): ICA-based artifact removal, adaptive filtering, z-score normalization
5. **Feature Extraction** (`analysis/brain_metrics.py`): Power spectral density, frequency band powers (delta, theta, alpha, beta, gamma)

### Machine Learning Architecture

**Problem**: Compare quantum and classical ML performance on EEG classification
**Solution**: Parallel implementation of quantum (QSVM, VQC) and classical (SVM, Random Forest) models
**Pros**: Direct performance comparison, quantum advantage demonstration
**Cons**: Quantum models have longer processing times

#### Quantum Models
- **QSVM** (`models/quantum_ml.py`): Quantum kernel-based SVM using PennyLane
  - 4-qubit quantum feature maps with angle and entanglement encoding
  - Quantum kernel matrix computation for similarity measurement
- **Enhanced QSVM** (`models/quantum_enhanced.py`): Deeper circuits with configurable entanglement
  - Hardware-efficient ansatz with 3-layer depth
  - Configurable entanglement patterns (circular, linear, full)
  - 4-8 qubit support

#### Classical Models
- **SVM** (`models/classical_ml.py`): Scikit-learn RBF kernel SVM
- **Random Forest**: Ensemble classifier for baseline comparison
- **Feature Reduction**: PCA for dimensionality reduction (95% variance threshold)

#### Model Configuration
- Centralized quantum configuration (`config/quantum_config.py`) for hyperparameters
- Adaptive PCA components based on data size
- Standardized preprocessing pipeline across all models

### Data Storage

**Problem**: Persist user data, analysis sessions, predictions, and activity logs
**Solution**: SQLite relational database with structured schema
**Rationale**: Lightweight, file-based, no separate server required

#### Database Schema (`database/db_manager.py`)
- **users**: User accounts with roles and authentication
- **sessions**: EEG analysis sessions with file metadata
- **predictions**: Model prediction results with accuracy metrics
- **brain_metrics**: Extracted frequency band powers and metrics
- **activity_logs**: Audit trail of user actions

#### Data Models (`database/models.py`)
- Dataclass-based models for type safety: User, Session, Prediction, BrainMetrics
- SQLite Row factory for dictionary-like access

### Security & Data Protection

**Problem**: Protect sensitive medical EEG data
**Solution**: Multi-layer security approach (`utils/security.py`)

- **Encryption**: Fernet symmetric encryption for data at rest
- **Password Hashing**: bcrypt with salt for authentication
- **Data Integrity**: SHA-256 hashing for verification
- **Access Control**: Role-based permissions for data export (`utils/data_export.py`)

### Batch Processing

**Problem**: Process multiple EEG files efficiently
**Solution**: Concurrent batch processor with thread-safe file I/O (`processing/batch_processor.py`)

- ThreadPoolExecutor for parallel file processing
- Thread-safe EDF reading with locks
- Configurable worker pool (default 4 workers)
- Individual error handling per file

### Reporting & Export

**Problem**: Generate shareable analysis reports
**Solution**: Multi-format export system with role-based filtering

- **PDF Reports** (`reports/pdf_generator.py`): FPDF-based comprehensive reports with session info, metrics, and predictions
- **Data Export** (`utils/data_export.py`): CSV/JSON export with permission checking
  - Researchers: metrics, predictions
  - Doctors: processed data, predictions, metrics
  - Admins: full access including raw data

### Real-Time Streaming Architecture (NEW)

**Problem**: Enable continuous EEG monitoring with live brain state analysis
**Solution**: Thread-safe ring buffer + sliding-window inference + EDF replay streaming
**Rationale**: Real-time insights require continuous data flow, low-latency processing, and smooth visualization

#### Streaming Components
- **Ring Buffer** (`streaming/ring_buffer.py`): Thread-safe circular buffer for 20 channels, 60-second capacity at 256Hz, supports window extraction and sample appending
- **EDF Replay Streamer** (`streaming/edf_replay.py`): Simulates live streaming from EDF files at configurable playback speeds (1x-10x), supports looping and callbacks
- **Brain State ML** (`streaming/brain_state_ml.py`): Real-time detection of cognitive load (theta/beta ratio), focus (engagement index, beta dominance), and anxiety (beta/alpha/theta patterns)
- **Inference Engine** (`streaming/inference_engine.py`): Sliding-window analysis (2s window, 0.5s hop), EWMA smoothing (alpha=0.3), background thread at ~2Hz, 10-minute history buffer
- **Predict Model UI** (`pages/2_Predict_Model.py`): Unified page with upload, analysis, and streaming
  - Left panel: File upload, brain metrics display, ML predictions, stream controls
  - Right panel: 20-channel blue line visualization (5s window), real-time gauges for cognitive load/focus/anxiety
  - Automatic workflow: Upload → Process → Stream with single button click

#### Data Flow
1. EDF Replay → Sample callback (256Hz)
2. Ring Buffer → Thread-safe storage (60s capacity)
3. Inference Engine → Sliding window analysis (every 0.5s)
4. Brain State ML → Cognitive metrics (0-100 scores)
5. UI Auto-refresh → Plotly updates (5Hz)

#### Brain State Detection
- **Cognitive Load**: Based on theta/beta ratio (higher theta = more load), calibrated baseline
- **Focus & Attention**: Engagement index (beta/(alpha+theta)), beta dominance, normalized to baseline
- **Anxiety**: High beta + low alpha + high theta patterns, multi-factor scoring
- **Calibration**: 30-second rest period for personalized baseline, per-user z-scoring

## External Dependencies

### Core ML & Quantum Computing
- **PennyLane**: Quantum machine learning framework for quantum circuits and kernels
- **NumPy**: Numerical computing and array operations
- **Scikit-learn**: Classical ML models (SVM, Random Forest), preprocessing, metrics

### Signal Processing
- **SciPy**: Signal filtering, spectral analysis, statistical functions
- **MNE**: EEG/MEG data processing (if used beyond PyEDFlib)
- **PyEDFlib**: EDF file format reading

### Web Framework & Visualization
- **Streamlit**: Web application framework and UI components
- **Plotly**: Interactive visualizations and charts
- **Matplotlib**: Static plotting (supplementary)

### Data Management
- **SQLite3**: Embedded relational database (Python standard library)
- **Pandas**: Data manipulation and CSV export
- **FPDF**: PDF report generation

### Security & Authentication
- **bcrypt**: Password hashing and verification
- **cryptography (Fernet)**: Symmetric encryption for data protection

### File Processing
- **pathlib**: Path manipulation (Python standard library)
- **tempfile**: Temporary file handling for uploads
- **pickle**: Python object serialization for encrypted data

### Development & Configuration
- No external configuration management system - uses Python class-based config (`config/quantum_config.py`)
- No external database - SQLite file-based storage
- No external API integrations - self-contained system