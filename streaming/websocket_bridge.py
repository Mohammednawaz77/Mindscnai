"""Bridge module to connect RingBuffer with WebSocket server"""
import threading

# Global ring buffer reference (shared across threads)
_ring_buffer = None
_buffer_lock = threading.Lock()

def set_ring_buffer(buffer):
    """Set the global ring buffer for WebSocket streaming"""
    global _ring_buffer
    with _buffer_lock:
        _ring_buffer = buffer

def get_ring_buffer():
    """Get the global ring buffer"""
    global _ring_buffer
    with _buffer_lock:
        return _ring_buffer

def clear_ring_buffer():
    """Clear the global ring buffer"""
    global _ring_buffer
    with _buffer_lock:
        _ring_buffer = None
