"""
WebSocket Integration for CoastVision Backend
Real-time alert broadcasting and mobile-dashboard sync
"""

from flask_socketio import SocketIO, emit, broadcast
from datetime import datetime
from typing import Dict, Optional

# Initialize SocketIO (will be called after Flask app creation)
socketio = None

# Track connected clients
CONNECTED_CLIENTS: Dict[str, dict] = {}


def init_socketio(app, cors_allowed_origins="*"):
    """Initialize Flask-SocketIO with the Flask app"""
    global socketio
    socketio = SocketIO(
        app,
        cors_allowed_origins=cors_allowed_origins,
        async_mode='threading',
        ping_timeout=60,
        ping_interval=25,
    )
    
    # Register event handlers
    @socketio.on('connect')
    def handle_connect(auth=None):
        """Client connects (web dashboard or mobile)"""
        client_type = 'unknown'
        user_id = None
        
        if auth:
            client_type = auth.get('client_type', 'unknown')
            user_id = auth.get('user_id')
        
        CONNECTED_CLIENTS[id] = {
            'connected_at': datetime.utcnow().isoformat(),
            'client_type': client_type,  # 'web' or 'mobile'
            'user_id': user_id,
        }
        print(f"[WS] Client connected: {id} (type={client_type})")
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Client disconnects"""
        if id in CONNECTED_CLIENTS:
            del CONNECTED_CLIENTS[id]
        print(f"[WS] Client disconnected: {id}")
    
    @socketio.on('ping')
    def handle_ping():
        """Client heartbeat"""
        emit('pong')
    
    return socketio


def broadcast_alert(alert_data: dict, skip_sid: Optional[str] = None):
    """
    Broadcast alert to ALL connected clients (dashboard + mobile apps)
    
    Args:
        alert_data: {
            'alert_id': str,
            'zone': str,
            'type': str,           # 'drowning', 'crowd', etc.
            'confidence': float,
            'timestamp': str,      # ISO format
            'image_url': str       # URL to alert image
        }
        skip_sid: Optional client ID to skip (usually sender)
    """
    if not socketio:
        return
    
    payload = {
        'alert_id': alert_data.get('alert_id'),
        'zone': alert_data.get('zone'),
        'type': alert_data.get('type'),
        'confidence': float(alert_data.get('confidence', 0)),
        'timestamp': alert_data.get('timestamp'),
        'image_url': alert_data.get('image_url'),
        'received_at': datetime.utcnow().isoformat(),
    }
    
    print(f"[WS] Broadcasting alert: {payload['alert_id']} ({payload['type']})")
    
    if skip_sid:
        socketio.emit('new_alert', payload, broadcast=True, skip_sid=skip_sid)
    else:
        socketio.emit('new_alert', payload, broadcast=True)


def broadcast_zone_update(zone_id: int, zone_data: dict):
    """
    Send zone update to all connected clients
    
    Args:
        zone_id: Zone ID
        zone_data: {
            'person_count': int,
            'status': str,         # 'normal', 'warning', 'critical'
            'detections': [...],   # List of detections
            'timestamp': str
        }
    """
    if not socketio:
        return
    
    payload = {
        'zone_id': zone_id,
        'person_count': zone_data.get('person_count', 0),
        'status': zone_data.get('status', 'normal'),
        'detections': zone_data.get('detections', []),
        'timestamp': zone_data.get('timestamp', datetime.utcnow().isoformat()),
    }
    
    socketio.emit(f'zone_update_{zone_id}', payload, broadcast=True)


def broadcast_lifeguard_response(alert_id: str, lifeguard_id: str, response_status: str, response_time: Optional[float] = None):
    """
    Notify all clients when lifeguard responds to alert
    
    Args:
        alert_id: Alert ID
        lifeguard_id: Lifeguard identifier
        response_status: 'acknowledged', 'en_route', 'resolved'
        response_time: Time to respond (seconds)
    """
    if not socketio:
        return
    
    payload = {
        'alert_id': alert_id,
        'lifeguard_id': lifeguard_id,
        'status': response_status,
        'response_time': response_time,
        'timestamp': datetime.utcnow().isoformat(),
    }
    
    print(f"[WS] Broadcasting response: {lifeguard_id} -> {response_status}")
    socketio.emit('lifeguard_response', payload, broadcast=True)


def broadcast_system_status(status_data: dict):
    """
    Broadcast system health status to all clients
    
    Args:
        status_data: {
            'gpu_usage': float,    # Percent
            'gpu_memory': float,   # Percent
            'inference_latency': float,  # ms
            'zones_active': int,
            'connected_lifeguards': int
        }
    """
    if not socketio:
        return
    
    socketio.emit('system_status', status_data, broadcast=True)


def emit_to_client(client_sid: str, event: str, data: dict):
    """Emit event to specific client"""
    if not socketio:
        return
    
    socketio.emit(event, data, to=client_sid)


def get_connected_clients_count() -> int:
    """Get number of connected clients"""
    return len(CONNECTED_CLIENTS)


def get_connected_clients_by_type(client_type: str) -> list:
    """Get list of connected clients by type ('web' or 'mobile')"""
    return [
        (sid, data)
        for sid, data in CONNECTED_CLIENTS.items()
        if data.get('client_type') == client_type
    ]
