"""Unified Predict Model Page - Upload, Analysis, and Live Streaming"""
import streamlit as st
import sys
from pathlib import Path
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.eeg_processor import EEGProcessor
from preprocessing.channel_selector import ChannelSelector
from models.quantum_ml import QuantumMLModels
from analysis.brain_metrics import BrainMetricsAnalyzer
from analysis.brain_state_classifier import BrainStateClassifier
from utils.security import SecurityManager
from streaming.ring_buffer import RingBuffer
from streaming.edf_replay import EDFReplayStreamer
from streaming.inference_engine import InferenceEngine
from streaming.websocket_bridge import set_ring_buffer, clear_ring_buffer
from components.websocket_signal_viewer import websocket_signal_viewer

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first")
    st.stop()

st.title("🔮 Predict Model - Real-Time EEG Analysis")

user = st.session_state.user
db_manager = st.session_state.db_manager

# Initialize streaming components once
if 'streaming_initialized' not in st.session_state:
    st.session_state.ring_buffer = RingBuffer(n_channels=20, buffer_seconds=60.0, sampling_rate=256.0)
    st.session_state.inference_engine = InferenceEngine(
        st.session_state.ring_buffer,
        window_seconds=2.0,
        hop_seconds=0.5
    )
    st.session_state.streamer = None
    st.session_state.streaming_initialized = True
    st.session_state.is_streaming = False

# TOP SECTION: Upload and Process Controls
st.markdown("### 📤 Upload & Process")

col_upload, col_process = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader("Choose EDF file", type=['edf'], help="Upload 64-channel EEG file")

with col_process:
    if uploaded_file is not None and 'file_processed' not in st.session_state:
        if st.button("🚀 Process & Start Streaming", type="primary", use_container_width=True):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.edf')
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.close()
            tmp_file_path = tmp_file.name
            
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("⏳ Reading EEG file...")
                progress_bar.progress(20)
                
                eeg_processor = EEGProcessor()
                channel_selector = ChannelSelector(target_channels=20)
                brain_state_classifier = BrainStateClassifier()
                
                data, channel_names, sampling_rate = eeg_processor.read_edf_file(tmp_file_path)
                progress_bar.progress(30)
                
                status_text.text("⏳ Preprocessing signals...")
                preprocessed_data = eeg_processor.preprocess_signals(data)
                progress_bar.progress(40)
                
                status_text.text("⏳ Selecting optimal channels...")
                selected_data, selected_indices, selected_names = channel_selector.select_optimal_channels(
                    preprocessed_data, channel_names, method='names'
                )
                progress_bar.progress(50)
                
                status_text.text("⏳ Computing brain metrics...")
                brain_analyzer = BrainMetricsAnalyzer(sampling_rate)
                
                channel_metrics_list = []
                for i in range(min(selected_data.shape[0], 20)):
                    ch_metrics = brain_analyzer.compute_channel_metrics(selected_data[i, :])
                    channel_metrics_list.append(ch_metrics)
                
                alpha_powers = [m.get('alpha', 0) for m in channel_metrics_list]
                beta_powers = [m.get('beta', 0) for m in channel_metrics_list]
                theta_powers = [m.get('theta', 0) for m in channel_metrics_list]
                delta_powers = [m.get('delta', 0) for m in channel_metrics_list]
                gamma_powers = [m.get('gamma', 0) for m in channel_metrics_list]
                
                alpha_power = float(np.median(alpha_powers))
                beta_power = float(np.median(beta_powers))
                theta_power = float(np.median(theta_powers))
                delta_power = float(np.median(delta_powers))
                gamma_power = float(np.median(gamma_powers))
                total_power = alpha_power + beta_power + theta_power + delta_power + gamma_power
                
                progress_bar.progress(65)
                
                status_text.text("⏳ Running scientific brain state analysis...")
                
                band_powers_dict = {
                    'alpha': alpha_power,
                    'beta': beta_power,
                    'theta': theta_power,
                    'delta': delta_power,
                    'gamma': gamma_power
                }
                
                brain_state, confidence, indices = brain_state_classifier.classify_brain_state(band_powers_dict)
                
                progress_bar.progress(85)
                
                brain_metrics = {
                    'alpha': alpha_power,
                    'beta': beta_power,
                    'theta': theta_power,
                    'delta': delta_power,
                    'gamma': gamma_power,
                    'alpha_relative': indices['alpha_relative'] * 100,
                    'beta_relative': indices['beta_relative'] * 100,
                    'theta_relative': indices['theta_relative'] * 100,
                    'delta_relative': indices['delta_relative'] * 100,
                    'gamma_relative': indices['gamma_relative'] * 100,
                    'brain_state': brain_state,
                    'engagement_index': indices['engagement_index'],
                    'relaxation_index': indices['relaxation_index'],
                    'drowsiness_index': indices['drowsiness_index'],
                    'vigilance_index': indices['vigilance_index'],
                    'mental_workload': indices['mental_workload'],
                    'classification_rationale': indices.get('classification_rationale', '')
                }
                
                session_id = db_manager.create_session(
                    user['id'],
                    uploaded_file.name,
                    len(channel_names),
                    len(selected_names)
                )
                
                db_manager.save_brain_metrics(
                    session_id,
                    alpha_power,
                    beta_power,
                    theta_power,
                    delta_power,
                    total_power,
                    brain_metrics
                )
                
                db_manager.save_prediction(
                    session_id,
                    'scientific',
                    'Brain State Classifier (Scientific)',
                    float(confidence),
                    brain_state,
                    0.05
                )
                
                st.session_state['current_session_id'] = session_id
                st.session_state['selected_data'] = selected_data
                st.session_state['selected_names'] = selected_names
                st.session_state['sampling_rate'] = sampling_rate
                st.session_state['brain_metrics'] = brain_metrics
                st.session_state['ml_prediction_name'] = brain_state
                st.session_state['ml_confidence'] = confidence
                st.session_state['scientific_indices'] = indices
                st.session_state['temp_file_path'] = tmp_file_path
                st.session_state['file_processed'] = True
                
                st.session_state.streamer = EDFReplayStreamer(
                    edf_file_path=tmp_file_path,
                    target_channels=20,
                    playback_speed=1.0
                )
                
                ring_buffer_ref = st.session_state.ring_buffer
                def on_sample(sample, timestamp):
                    ring_buffer_ref.append(sample)
                
                st.session_state.streamer.start_stream(callback=on_sample, loop=True)
                st.session_state.inference_engine.start()
                st.session_state.is_streaming = True
                
                # Connect ring buffer to WebSocket server
                set_ring_buffer(st.session_state.ring_buffer)
                
                progress_bar.progress(100)
                status_text.text("")
                st.success("✅ Analysis complete! Streaming started.")
                db_manager.log_activity(user['id'], 'analysis', f"Processed {uploaded_file.name}")
                
                st.rerun()
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                db_manager.log_activity(user['id'], 'error', f"Processing failed: {str(e)}")
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

st.markdown("---")

# MIDDLE SECTION: Analysis Results and Controls
if 'brain_metrics' in st.session_state:
    
    # Split into two columns: Results and Controls
    results_col, controls_col = st.columns([2, 1])
    
    with results_col:
        st.markdown("### 📊 Analysis Results")
        
        metrics = st.session_state['brain_metrics']
        
        # Band powers in compact format
        band_col1, band_col2, band_col3, band_col4, band_col5 = st.columns(5)
        with band_col1:
            st.metric("Alpha", f"{metrics.get('alpha', 0):.2f}", help="8-13 Hz")
        with band_col2:
            st.metric("Beta", f"{metrics.get('beta', 0):.2f}", help="13-30 Hz")
        with band_col3:
            st.metric("Theta", f"{metrics.get('theta', 0):.2f}", help="4-8 Hz")
        with band_col4:
            st.metric("Delta", f"{metrics.get('delta', 0):.2f}", help="0.5-4 Hz")
        with band_col5:
            st.metric("Gamma", f"{metrics.get('gamma', 0):.2f}", help="30-100 Hz")
        
        st.markdown("**🧬 Brain State:**")
        pred_state = st.session_state.get('ml_prediction_name', 'Unknown')
        confidence = st.session_state.get('ml_confidence', 0)
        st.info(f"**{pred_state}** (Confidence: {confidence*100:.1f}%)")
        
        if "Deep Sleep" in pred_state:
            st.caption("ℹ️ Delta waves dominant - Stage 3-4 NREM sleep pattern")
    
    with controls_col:
        st.markdown("### 🎮 Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.is_streaming:
                if st.button("⏸️ Stop", use_container_width=True, key="stop_btn"):
                    if st.session_state.streamer:
                        st.session_state.streamer.stop_stream()
                        st.session_state.inference_engine.stop()
                        st.session_state.is_streaming = False
                        
                        # Disconnect ring buffer from WebSocket server
                        clear_ring_buffer()
                        
                        if 'temp_file_path' in st.session_state:
                            temp_path = st.session_state['temp_file_path']
                            if os.path.exists(temp_path):
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                            del st.session_state['temp_file_path']
                        if 'file_processed' in st.session_state:
                            del st.session_state['file_processed']
                        st.success("Stopped")
                        st.rerun()
            else:
                st.button("⏸️ Stop", disabled=True, use_container_width=True)
        
        with col2:
            if st.session_state.is_streaming:
                if st.button("🎯 Calibrate", use_container_width=True, key="cal_btn"):
                    buffer_stats = st.session_state.ring_buffer.get_stats()
                    seconds_available = buffer_stats['current_samples'] / buffer_stats['sampling_rate']
                    
                    if seconds_available >= 30:
                        with st.spinner("Calibrating..."):
                            success = st.session_state.inference_engine.calibrate()
                        if success:
                            st.success("✅ Calibrated")
                        else:
                            st.error("❌ Failed")
                    else:
                        st.warning(f"⏳ Need {30-seconds_available:.1f}s more")
            else:
                st.button("🎯 Calibrate", disabled=True, use_container_width=True)
        
        # Status
        buffer_stats = st.session_state.ring_buffer.get_stats()
        st.metric("Status", "🟢 Streaming" if st.session_state.is_streaming else "🔴 Stopped")
        st.metric("Buffer", f"{buffer_stats['fill_percentage']:.1f}%")
    
    # Real-time brain states gauges
    results = st.session_state.inference_engine.get_latest_results()
    
    if results:
        st.markdown("---")
        st.markdown("### 🧠 Real-Time Brain States")
        
        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
        
        with gauge_col1:
            cog_load = results['cognitive_load']
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cog_load.get('smoothed_score', cog_load['score']),
                title={'text': "Cognitive Load"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2196F3"},
                    'steps': [
                        {'range': [0, 30], 'color': "#E8F5E9"},
                        {'range': [30, 70], 'color': "#FFF9C4"},
                        {'range': [70, 100], 'color': "#FFCDD2"}
                    ]
                }
            ))
            fig_g1.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_g1, use_container_width=True)
        
        with gauge_col2:
            focus = results['focus']
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=focus.get('smoothed_score', focus['score']),
                title={'text': "Focus"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4CAF50"},
                    'steps': [
                        {'range': [0, 30], 'color': "#FFCDD2"},
                        {'range': [30, 70], 'color': "#FFF9C4"},
                        {'range': [70, 100], 'color': "#E8F5E9"}
                    ]
                }
            ))
            fig_g2.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_g2, use_container_width=True)
        
        with gauge_col3:
            anxiety = results['anxiety']
            fig_g3 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=anxiety.get('smoothed_score', anxiety['score']),
                title={'text': "Anxiety"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF9800"},
                    'steps': [
                        {'range': [0, 30], 'color': "#E8F5E9"},
                        {'range': [30, 60], 'color': "#FFF9C4"},
                        {'range': [60, 100], 'color': "#FFCDD2"}
                    ]
                }
            ))
            fig_g3.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_g3, use_container_width=True)

st.markdown("---")

# BOTTOM SECTION: Continuous EEG Signal Display (ALL 20 CHANNELS) - WebSocket Version
st.markdown("### 🌊 Continuous Signal Display - Real-Time EEG (5-second window)")

# Get channel names for WebSocket viewer
channel_names = st.session_state.get('selected_names', [f"Ch{i+1}" for i in range(20)])

if st.session_state.is_streaming:
    # Render WebSocket-based signal viewer (NO FLICKERING!)
    websocket_signal_viewer(channel_names, height=800)
    st.caption("📡 Real-time continuous EEG streaming via WebSocket - All 20 channels displayed - NO FLICKERING")
else:
    st.info("⏳ Waiting for signal data... Upload a file and start streaming to see live EEG signals")
