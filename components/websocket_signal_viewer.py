"""WebSocket-based EEG Signal Viewer Component for Streamlit"""
import streamlit as st
import streamlit.components.v1 as components

def websocket_signal_viewer(channel_names, height=800):
    """
    Render WebSocket-based EEG signal viewer
    
    Args:
        channel_names: List of channel names to display
        height: Height of the component in pixels
    """
    
    # HTML/JS component with WebSocket connection and Plotly visualization
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #0a0a0a;
                font-family: sans-serif;
            }}
            #chart {{
                width: 100%;
                height: 100%;
            }}
            #status {{
                position: absolute;
                top: 10px;
                right: 10px;
                color: #00D9FF;
                font-size: 12px;
                padding: 5px 10px;
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 4px;
            }}
            .connected {{
                color: #4CAF50;
            }}
            .disconnected {{
                color: #f44336;
            }}
        </style>
    </head>
    <body>
        <div id="status">Connecting...</div>
        <div id="chart"></div>
        
        <script>
            const channelNames = {channel_names};
            const nChannels = channelNames.length;
            let ws = null;
            let reconnectInterval = null;
            
            // Initialize Plotly chart
            const layout = {{
                height: {height},
                margin: {{ l: 80, r: 20, t: 30, b: 50 }},
                paper_bgcolor: '#0a0a0a',
                plot_bgcolor: '#1a1a1a',
                xaxis: {{
                    title: 'Time (seconds)',
                    gridcolor: '#2a2a2a',
                    showgrid: true,
                    gridwidth: 1,
                    color: '#00D9FF',
                    range: [0, 5],
                    dtick: 0.5
                }},
                yaxis: {{
                    showticklabels: false,
                    gridcolor: '#2a2a2a',
                    showgrid: true,
                    gridwidth: 1,
                    zeroline: false
                }},
                hovermode: false,
                font: {{ color: '#00D9FF' }},
                showlegend: false
            }};
            
            // Create initial empty traces
            const traces = [];
            for (let i = 0; i < nChannels; i++) {{
                traces.push({{
                    x: [],
                    y: [],
                    mode: 'lines',
                    line: {{ color: '#00D9FF', width: 1.2 }},
                    hoverinfo: 'skip'
                }});
            }}
            
            Plotly.newPlot('chart', traces, layout, {{
                displayModeBar: false,
                responsive: true
            }});
            
            // Add channel labels
            const annotations = [];
            for (let i = 0; i < nChannels; i++) {{
                const offset = (nChannels - i - 1) * 150;
                annotations.push({{
                    x: -0.15,
                    y: offset,
                    text: channelNames[i],
                    showarrow: false,
                    xref: 'x',
                    yref: 'y',
                    font: {{ color: '#00D9FF', size: 10 }},
                    xanchor: 'right'
                }});
            }}
            Plotly.relayout('chart', {{ annotations: annotations }});
            
            function connectWebSocket() {{
                // Connect to WebSocket server on port 8000
                const wsUrl = `ws://${{window.location.hostname}}:8000/ws/eeg`;
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {{
                    console.log('WebSocket connected');
                    document.getElementById('status').textContent = '🟢 Connected';
                    document.getElementById('status').className = 'connected';
                    
                    // Send ping every 30 seconds to keep connection alive
                    setInterval(() => {{
                        if (ws.readyState === WebSocket.OPEN) {{
                            ws.send('ping');
                        }}
                    }}, 30000);
                }};
                
                ws.onmessage = (event) => {{
                    try {{
                        const message = JSON.parse(event.data);
                        
                        if (message.type === 'signal_update') {{
                            const signalData = message.data;
                            const nSamples = message.n_samples;
                            const samplingRate = message.sampling_rate;
                            
                            // Create time axis
                            const timeAxis = Array.from({{length: nSamples}}, (_, i) => i / samplingRate);
                            
                            // Update traces
                            const updates = [];
                            for (let i = 0; i < Math.min(nChannels, signalData.length); i++) {{
                                const offset = (nChannels - i - 1) * 150;
                                const traceData = signalData[i].map(v => v + offset);
                                updates.push({{
                                    x: [timeAxis],
                                    y: [traceData]
                                }});
                            }}
                            
                            // Efficient update using Plotly.react
                            Plotly.react('chart', updates.map((u, i) => ({{
                                ...traces[i],
                                x: u.x[0],
                                y: u.y[0]
                            }})), layout);
                        }}
                    }} catch (e) {{
                        console.error('Error processing message:', e);
                    }}
                }};
                
                ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                }};
                
                ws.onclose = () => {{
                    console.log('WebSocket disconnected');
                    document.getElementById('status').textContent = '🔴 Disconnected';
                    document.getElementById('status').className = 'disconnected';
                    
                    // Attempt reconnection after 2 seconds
                    setTimeout(connectWebSocket, 2000);
                }};
            }}
            
            // Initial connection
            connectWebSocket();
        </script>
    </body>
    </html>
    """
    
    # Render the component
    components.html(html_code, height=height)
