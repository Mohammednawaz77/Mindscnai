import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
import io
import base64

class SignalVisualizer:
    """Visualize EEG signals and analysis results"""
    
    def __init__(self):
        plt.style.use('default')
    
    def plot_raw_signals(self, data: np.ndarray, channel_names: List[str], 
                        sampling_rate: float = 256, duration: float = 10) -> go.Figure:
        """Plot raw EEG signals using Plotly"""
        
        n_channels = len(channel_names)
        samples_to_plot = int(duration * sampling_rate)
        time = np.arange(samples_to_plot) / sampling_rate
        
        fig = make_subplots(
            rows=n_channels, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=channel_names
        )
        
        for i, channel_name in enumerate(channel_names):
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=data[i, :samples_to_plot],
                    mode='lines',
                    name=channel_name,
                    line=dict(width=1)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=200*n_channels,
            title_text="20-Channel EEG Signals",
            showlegend=False,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time (s)", row=n_channels, col=1)
        
        return fig
    
    def plot_power_spectrum(self, data: np.ndarray, sampling_rate: float = 256) -> go.Figure:
        """Plot power spectral density"""
        from scipy import signal as sp_signal
        
        # Average across channels
        avg_data = np.mean(data, axis=0)
        
        # Compute PSD
        freqs, psd = sp_signal.welch(avg_data, fs=sampling_rate, nperseg=sampling_rate*2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=freqs,
            y=10 * np.log10(psd),
            mode='lines',
            name='PSD',
            line=dict(color='blue', width=2)
        ))
        
        # Add band regions
        bands = {
            'Delta': (0.5, 4, 'rgba(128,0,128,0.1)'),
            'Theta': (4, 8, 'rgba(0,0,255,0.1)'),
            'Alpha': (8, 13, 'rgba(0,255,0,0.1)'),
            'Beta': (13, 30, 'rgba(255,165,0,0.1)'),
            'Gamma': (30, 50, 'rgba(255,0,0,0.1)')
        }
        
        for band_name, (low, high, color) in bands.items():
            fig.add_vrect(
                x0=low, x1=high,
                fillcolor=color,
                layer="below",
                line_width=0,
                annotation_text=band_name,
                annotation_position="top left"
            )
        
        fig.update_layout(
            title="Power Spectral Density",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB)",
            height=400,
            hovermode='x'
        )
        
        return fig
    
    def plot_band_powers(self, metrics: Dict) -> go.Figure:
        """Plot frequency band powers"""
        
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        powers = [metrics[f'{band}_power'] for band in bands]
        
        fig = go.Figure(data=[
            go.Bar(
                x=bands,
                y=powers,
                marker_color=['purple', 'blue', 'green', 'orange', 'red'],
                text=[f'{p:.2f}' for p in powers],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Frequency Band Powers",
            xaxis_title="Frequency Band",
            yaxis_title="Power (μV²)",
            height=400
        )
        
        return fig
    
    def plot_relative_powers(self, metrics: Dict) -> go.Figure:
        """Plot relative band powers as pie chart"""
        
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        values = [
            metrics['delta_relative'],
            metrics['theta_relative'],
            metrics['alpha_relative'],
            metrics['beta_relative'],
            metrics['gamma_relative']
        ]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=bands,
                values=values,
                marker=dict(colors=['purple', 'blue', 'green', 'orange', 'red']),
                textinfo='label+percent'
            )
        ])
        
        fig.update_layout(
            title="Relative Band Power Distribution",
            height=400
        )
        
        return fig
    
    def plot_model_comparison(self, predictions_dict: Dict) -> go.Figure:
        """Compare model accuracies"""
        
        models = list(predictions_dict.keys())
        accuracies = [predictions_dict[m]['accuracy'] for m in models]
        times = [predictions_dict[m]['processing_time'] for m in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Accuracy', 'Processing Time'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=accuracies,
                name='Accuracy',
                marker_color='steelblue',
                text=[f'{a:.2%}' for a in accuracies],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=times,
                name='Time (s)',
                marker_color='coral',
                text=[f'{t:.3f}s' for t in times],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
        
        return fig
    
    def create_matplotlib_figure(self, data: np.ndarray, channel_names: List[str]) -> str:
        """Create matplotlib figure and return as base64 string"""
        
        fig, axes = plt.subplots(len(channel_names), 1, figsize=(12, 2*len(channel_names)))
        
        if len(channel_names) == 1:
            axes = [axes]
        
        for i, (ax, channel_name) in enumerate(zip(axes, channel_names)):
            ax.plot(data[i, :1000], linewidth=0.5)
            ax.set_ylabel(channel_name)
            ax.set_xlim(0, 1000)
            ax.grid(True, alpha=0.3)
            
            if i == len(channel_names) - 1:
                ax.set_xlabel('Samples')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
