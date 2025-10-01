"""FastAPI WebSocket Server for Real-Time EEG Signal Streaming"""
import asyncio
import json
import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Set
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from streaming.websocket_bridge import get_ring_buffer

app = FastAPI()

# CORS middleware to allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global registry for connected WebSocket clients
connected_clients: Set[WebSocket] = set()

@app.websocket("/ws/eeg")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for EEG signal streaming"""
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"WebSocket client connected. Total clients: {len(connected_clients)}")
    
    try:
        # Keep connection alive - just wait for disconnect
        while True:
            try:
                # Non-blocking receive with timeout - handle pings if sent
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # No message received, continue - this is normal
                pass
            except WebSocketDisconnect:
                break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
            print(f"WebSocket client disconnected. Total clients: {len(connected_clients)}")

async def broadcast_signals():
    """Continuously broadcast EEG signals to all connected clients"""
    while True:
        ring_buffer = get_ring_buffer()
        if ring_buffer and len(connected_clients) > 0:
            try:
                # Get latest 5 seconds of data (1280 samples at 256 Hz)
                signal_data = ring_buffer.get_latest(n_samples=1280)
                
                if signal_data.shape[1] > 0:
                    # Convert to JSON-serializable format
                    payload = {
                        "type": "signal_update",
                        "data": signal_data.tolist(),
                        "n_channels": signal_data.shape[0],
                        "n_samples": signal_data.shape[1],
                        "sampling_rate": 256
                    }
                    
                    # Broadcast to all connected clients
                    disconnected = set()
                    for client in connected_clients:
                        try:
                            await client.send_json(payload)
                        except Exception as e:
                            print(f"Error sending to client: {e}")
                            disconnected.add(client)
                    
                    # Remove disconnected clients
                    for client in disconnected:
                        connected_clients.discard(client)
                
            except Exception as e:
                print(f"Broadcast error: {e}")
        
        # Broadcast at 5 Hz (every 200ms)
        await asyncio.sleep(0.2)

@app.on_event("startup")
async def startup_event():
    """Start background task for broadcasting"""
    asyncio.create_task(broadcast_signals())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "connected_clients": len(connected_clients),
        "ring_buffer_active": get_ring_buffer() is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
