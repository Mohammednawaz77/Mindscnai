"""Model Comparison Dashboard - Quantum vs Classical ML"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("⚠️ Please login from the main page")
    st.stop()

st.title("🔬 Quantum vs Classical ML Comparison")
st.markdown("### Comprehensive Model Performance Analysis")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Metrics", "⚡ Processing Speed", "🎯 Accuracy Analysis", "🔍 Detailed Comparison"])

# Sample data structure - will be populated from session state
if 'model_results' not in st.session_state:
    st.session_state.model_results = {
        'quantum_models': [
            {'name': 'QSVM', 'accuracy': 0.8571, 'precision': 0.85, 'recall': 0.86, 'f1': 0.855, 
             'processing_time': 0.891, 'n_qubits': 4, 'type': 'quantum'},
            {'name': 'Enhanced QSVM', 'accuracy': 0.9286, 'precision': 0.92, 'recall': 0.93, 'f1': 0.925,
             'processing_time': 1.031, 'n_qubits': 4, 'type': 'quantum'},
        ],
        'classical_models': [
            {'name': 'SVM', 'accuracy': 0.6667, 'precision': 0.65, 'recall': 0.68, 'f1': 0.665,
             'processing_time': 0.042, 'type': 'classical'},
            {'name': 'Random Forest', 'accuracy': 0.6667, 'precision': 0.64, 'recall': 0.69, 'f1': 0.665,
             'processing_time': 0.018, 'type': 'classical'},
        ]
    }

model_results = st.session_state.model_results

# Tab 1: Performance Metrics
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Accuracy Comparison")
        
        # Prepare data
        all_models = model_results['quantum_models'] + model_results['classical_models']
        df = pd.DataFrame(all_models)
        
        # Create bar chart
        fig = go.Figure()
        
        quantum_df = df[df['type'] == 'quantum']
        classical_df = df[df['type'] == 'classical']
        
        fig.add_trace(go.Bar(
            x=quantum_df['name'],
            y=quantum_df['accuracy'] * 100,
            name='Quantum Models',
            marker_color='#8B5CF6',
            text=[f"{acc:.1f}%" for acc in quantum_df['accuracy'] * 100],
            textposition='auto',
        ))
        
        fig.add_trace(go.Bar(
            x=classical_df['name'],
            y=classical_df['accuracy'] * 100,
            name='Classical Models',
            marker_color='#10B981',
            text=[f"{acc:.1f}%" for acc in classical_df['accuracy'] * 100],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy (%)",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 F1-Score Comparison")
        
        # F1 Score comparison
        fig_f1 = go.Figure()
        
        fig_f1.add_trace(go.Bar(
            x=quantum_df['name'],
            y=quantum_df['f1'] * 100,
            name='Quantum Models',
            marker_color='#8B5CF6',
            text=[f"{f1:.1f}%" for f1 in quantum_df['f1'] * 100],
            textposition='auto',
        ))
        
        fig_f1.add_trace(go.Bar(
            x=classical_df['name'],
            y=classical_df['f1'] * 100,
            name='Classical Models',
            marker_color='#10B981',
            text=[f"{f1:.1f}%" for f1 in classical_df['f1'] * 100],
            textposition='auto',
        ))
        
        fig_f1.update_layout(
            title="Model F1-Score Comparison",
            xaxis_title="Model",
            yaxis_title="F1-Score (%)",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # Precision-Recall comparison
    st.subheader("🎯 Precision vs Recall")
    
    fig_pr = go.Figure()
    
    for model in all_models:
        color = '#8B5CF6' if model['type'] == 'quantum' else '#10B981'
        fig_pr.add_trace(go.Scatter(
            x=[model['precision']],
            y=[model['recall']],
            mode='markers+text',
            name=model['name'],
            marker=dict(size=15, color=color),
            text=[model['name']],
            textposition='top center'
        ))
    
    fig_pr.update_layout(
        title="Precision vs Recall Analysis",
        xaxis_title="Precision",
        yaxis_title="Recall",
        height=400
    )
    
    st.plotly_chart(fig_pr, use_container_width=True)

# Tab 2: Processing Speed
with tab2:
    st.subheader("⚡ Processing Time Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing time bar chart
        fig_time = go.Figure()
        
        fig_time.add_trace(go.Bar(
            x=quantum_df['name'],
            y=quantum_df['processing_time'],
            name='Quantum Models',
            marker_color='#8B5CF6',
            text=[f"{t:.3f}s" for t in quantum_df['processing_time']],
            textposition='auto',
        ))
        
        fig_time.add_trace(go.Bar(
            x=classical_df['name'],
            y=classical_df['processing_time'],
            name='Classical Models',
            marker_color='#10B981',
            text=[f"{t:.3f}s" for t in classical_df['processing_time']],
            textposition='auto',
        ))
        
        fig_time.update_layout(
            title="Processing Time Comparison",
            xaxis_title="Model",
            yaxis_title="Time (seconds)",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Speed efficiency (accuracy per second)
        fastest_idx = df['processing_time'].argmin()
        st.metric("⚡ Fastest Model", 
                  df.iloc[fastest_idx]['name'],
                  f"{df.iloc[fastest_idx]['processing_time']:.3f}s")
        
        st.metric("🏆 Most Accurate", 
                  df.iloc[df['accuracy'].argmax()]['name'],
                  f"{df['accuracy'].max():.2%}")
        
        # Efficiency metric: accuracy/time
        df['efficiency'] = df['accuracy'] / df['processing_time']
        best_efficiency = df.iloc[df['efficiency'].argmax()]
        st.metric("⚖️ Best Efficiency (Acc/Time)", 
                  best_efficiency['name'],
                  f"{best_efficiency['efficiency']:.2f}")

# Tab 3: Accuracy Analysis
with tab3:
    st.subheader("🎯 Detailed Accuracy Analysis")
    
    # Radar chart for model comparison
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig_radar = go.Figure()
    
    for model in all_models:
        fig_radar.add_trace(go.Scatterpolar(
            r=[model['accuracy']*100, model['precision']*100, 
               model['recall']*100, model['f1']*100],
            theta=categories,
            fill='toself',
            name=model['name']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Multi-Metric Model Comparison (Radar Chart)",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Performance table
    st.subheader("📊 Performance Summary Table")
    
    summary_df = pd.DataFrame(all_models)
    summary_df = summary_df[['name', 'type', 'accuracy', 'precision', 'recall', 'f1', 'processing_time']]
    summary_df.columns = ['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time (s)']
    
    # Format percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2%}")  # type: ignore
    
    summary_df['Time (s)'] = summary_df['Time (s)'].apply(lambda x: f"{x:.3f}")  # type: ignore
    
    st.dataframe(summary_df, use_container_width=True)

# Tab 4: Detailed Comparison
with tab4:
    st.subheader("🔍 Quantum Advantage Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate quantum advantage
    best_quantum = max([m['accuracy'] for m in model_results['quantum_models']])
    best_classical = max([m['accuracy'] for m in model_results['classical_models']])
    quantum_advantage = (best_quantum - best_classical) / (best_classical + 1e-10) * 100
    
    with col1:
        st.metric("🚀 Quantum Advantage", 
                  f"{quantum_advantage:.1f}%",
                  "Higher accuracy than classical")
    
    with col2:
        avg_quantum_time = np.mean([m['processing_time'] for m in model_results['quantum_models']])
        avg_classical_time = np.mean([m['processing_time'] for m in model_results['classical_models']])
        st.metric("⏱️ Avg Quantum Time", 
                  f"{avg_quantum_time:.3f}s",
                  f"+{(avg_quantum_time/avg_classical_time - 1)*100:.0f}% vs classical")
    
    with col3:
        st.metric("🎯 Best Overall Model",
                  df.iloc[df['accuracy'].argmax()]['name'],
                  f"{df['accuracy'].max():.2%} accuracy")
    
    # Quantum model details
    st.subheader("⚛️ Quantum Model Configuration")
    
    quantum_config_data = []
    for qm in model_results['quantum_models']:
        quantum_config_data.append({
            'Model': qm['name'],
            'Qubits': qm.get('n_qubits', 'N/A'),
            'Accuracy': f"{qm['accuracy']:.2%}",
            'Time': f"{qm['processing_time']:.3f}s",
            'Type': 'Quantum Kernel SVM'
        })
    
    st.dataframe(pd.DataFrame(quantum_config_data), use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Model Selection Recommendations")
    
    st.info("""
    **When to use Quantum Models:**
    - High-accuracy requirements (medical diagnosis, critical applications)
    - Complex pattern recognition in high-dimensional data
    - When computational resources allow for longer processing times
    
    **When to use Classical Models:**
    - Real-time predictions needed (< 50ms response time)
    - Resource-constrained environments
    - When accuracy difference is acceptable for the use case
    """)
    
    # Statistical significance
    st.subheader("📈 Statistical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Quantum Models:**")
        st.write(f"- Average Accuracy: {np.mean([m['accuracy'] for m in model_results['quantum_models']]):.2%}")
        st.write(f"- Std Dev: {np.std([m['accuracy'] for m in model_results['quantum_models']]):.4f}")
        st.write(f"- Average Time: {avg_quantum_time:.3f}s")
    
    with col2:
        st.write("**Classical Models:**")
        st.write(f"- Average Accuracy: {np.mean([m['accuracy'] for m in model_results['classical_models']]):.2%}")
        st.write(f"- Std Dev: {np.std([m['accuracy'] for m in model_results['classical_models']]):.4f}")
        st.write(f"- Average Time: {avg_classical_time:.3f}s")

# Footer
st.markdown("---")
st.caption(f"👤 Logged in as: {st.session_state.get('username', 'Unknown')} | Role: {st.session_state.get('role', 'Unknown')}")
