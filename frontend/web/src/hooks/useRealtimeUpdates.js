/**
 * useRealtimeUpdates: React Hook for WebSocket Real-Time Updates
 * Syncs alerts, zone updates, and lifeguard responses in real-time
 */

import { useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';

export const useRealtimeUpdates = (
  onNewAlert = null,
  onZoneUpdate = null,
  onLifeguardResponse = null,
  onSystemStatus = null,
  backendUrl = 'http://127.0.0.1:8000'
) => {
  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  
  // Store callbacks in refs to avoid re-initialization on every render
  const callbacksRef = useRef({
    onNewAlert,
    onZoneUpdate,
    onLifeguardResponse,
    onSystemStatus,
  });
  
  // Update callbacks ref when they change
  useEffect(() => {
    callbacksRef.current = {
      onNewAlert,
      onZoneUpdate,
      onLifeguardResponse,
      onSystemStatus,
    };
  }, [onNewAlert, onZoneUpdate, onLifeguardResponse, onSystemStatus]);

  // Initialize WebSocket connection - only once per component mount
  useEffect(() => {
    if (socketRef.current?.connected) {
      return; // Already connected
    }

    console.log('[WebSocket] Connecting to', backendUrl);

    socketRef.current = io(backendUrl, {
      auth: { 
        client_type: 'web',
        user_id: localStorage.getItem('user_id') || 'admin',
      },
      reconnection: true,
      reconnectionDelay: 2000,
      reconnectionDelayMax: 10000,
      reconnectionAttempts: Infinity,
      transports: ['websocket'],
      forceNew: false,
      multiplex: true,
    });

    // Connection events (silent - no sounds)
    socketRef.current.on('connect', () => {
      console.log('[WebSocket] Connected to backend');
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    });

    socketRef.current.on('disconnect', () => {
      console.log('[WebSocket] Disconnected from backend');
    });

    socketRef.current.on('connect_error', (error) => {
      console.error('[WebSocket] Connection error:', error);
    });

    // Alert events
    socketRef.current.on('new_alert', (alert) => {
      console.log('[WebSocket] New alert received:', alert);
      if (callbacksRef.current.onNewAlert) {
        callbacksRef.current.onNewAlert(alert);
      }
      // MUTED: Alert sound disabled
      // playAlertSound();
      // MUTED: Desktop notifications disabled
      // showDesktopNotification(alert);
    });

    // Zone update events
    socketRef.current.on('zone_update', (data) => {
      console.log('[WebSocket] Zone update:', data);
      if (callbacksRef.current.onZoneUpdate) {
        callbacksRef.current.onZoneUpdate(data);
      }
    });

    // Zone-specific updates (dynamic event name)
    for (let zid = 0; zid < 20; zid++) {
      socketRef.current.on(`zone_update_${zid}`, (data) => {
        if (callbacksRef.current.onZoneUpdate) {
          callbacksRef.current.onZoneUpdate(data);
        }
      });
    }

    // Lifeguard response events
    socketRef.current.on('lifeguard_response', (response) => {
      console.log('[WebSocket] Lifeguard response:', response);
      if (callbacksRef.current.onLifeguardResponse) {
        callbacksRef.current.onLifeguardResponse(response);
      }
    });

    // System status events
    socketRef.current.on('system_status', (status) => {
      console.log('[WebSocket] System status:', status);
      if (callbacksRef.current.onSystemStatus) {
        callbacksRef.current.onSystemStatus(status);
      }
    });

    // Cleanup on unmount ONLY
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, [backendUrl]); // Only depend on backendUrl

  // Send heartbeat ping
  useEffect(() => {
    const interval = setInterval(() => {
      if (socketRef.current?.connected) {
        socketRef.current.emit('ping');
      }
    }, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Utility: Play alert sound (DISABLED - causes "ting" sounds)
  const playAlertSound = useCallback(() => {
    // Alert sound disabled
    return;
  }, []);

  // Utility: Show browser notification (DISABLED to prevent sounds)
  const showDesktopNotification = useCallback((alert) => {
    // Notifications disabled to prevent "ting" sounds
    return;
  }, []);

  // Return utility functions
  return {
    socket: socketRef.current,
    isConnected: socketRef.current?.connected || false,
    emit: (event, data) => {
      if (socketRef.current?.connected) {
        socketRef.current.emit(event, data);
      }
    },
    playAlertSound,
    showDesktopNotification,
  };
};

export default useRealtimeUpdates;
