"""Start both Streamlit and WebSocket servers"""
import subprocess
import sys
import time
import signal
import os

# Store process references
processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down servers...")
    for p in processes:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("🚀 Starting QuantumBCI Servers...")
    
    # Start WebSocket server on port 8000
    print("📡 Starting WebSocket server on port 8000...")
    ws_process = subprocess.Popen(
        ["python", "websocket_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes.append(ws_process)
    time.sleep(2)  # Give WebSocket server time to start
    
    # Start Streamlit on port 5000
    print("🎨 Starting Streamlit on port 5000...")
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes.append(streamlit_process)
    
    print("✅ Both servers started!")
    print("   - Streamlit UI: http://0.0.0.0:5000")
    print("   - WebSocket: ws://0.0.0.0:8000/ws/eeg")
    print("\nPress Ctrl+C to stop all servers")
    
    # Keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)
