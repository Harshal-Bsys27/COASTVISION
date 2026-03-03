import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  Vibration,
  StatusBar,
  SafeAreaView,
  Platform,
  AppState,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

// ============ CONFIGURATION ============
// Change this to your computer's IP address on your WiFi network
// Find it by running 'ipconfig' in PowerShell and looking for IPv4 Address
const API_BASE = 'http://10.202.83.183:8000'; // <-- CHANGE THIS TO YOUR PC's IP

// ============ MAIN APP ============
export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [lifeguardId, setLifeguardId] = useState(null);
  const [lifeguardName, setLifeguardName] = useState('');
  const [assignedZones, setAssignedZones] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  
  // Login form state
  const [nameInput, setNameInput] = useState('');
  const [phoneInput, setPhoneInput] = useState('');
  
  const appState = useRef(AppState.currentState);
  const pollIntervalRef = useRef(null);

  // Load saved session on app start
  useEffect(() => {
    loadSavedSession();
  }, []);

  // Setup polling when logged in
  useEffect(() => {
    if (isLoggedIn && lifeguardId) {
      startPolling();
      sendHeartbeat();
    }
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [isLoggedIn, lifeguardId]);

  // Handle app state changes (background/foreground)
  useEffect(() => {
    const subscription = AppState.addEventListener('change', nextAppState => {
      if (appState.current.match(/inactive|background/) && nextAppState === 'active') {
        // App came to foreground
        if (isLoggedIn && lifeguardId) {
          loadAlerts();
          sendHeartbeat();
        }
      }
      appState.current = nextAppState;
    });
    return () => subscription?.remove();
  }, [isLoggedIn, lifeguardId]);

  const loadSavedSession = async () => {
    try {
      const savedId = await AsyncStorage.getItem('lifeguardId');
      const savedName = await AsyncStorage.getItem('lifeguardName');
      if (savedId && savedName) {
        setLifeguardId(savedId);
        setLifeguardName(savedName);
        setIsLoggedIn(true);
      }
    } catch (error) {
      console.error('Error loading session:', error);
    }
  };

  const saveSession = async (id, name) => {
    try {
      await AsyncStorage.setItem('lifeguardId', id);
      await AsyncStorage.setItem('lifeguardName', name);
    } catch (error) {
      console.error('Error saving session:', error);
    }
  };

  const clearSession = async () => {
    try {
      await AsyncStorage.removeItem('lifeguardId');
      await AsyncStorage.removeItem('lifeguardName');
    } catch (error) {
      console.error('Error clearing session:', error);
    }
  };

  const handleLogin = async () => {
    if (!nameInput.trim()) {
      Alert.alert('Error', 'Please enter your name');
      return;
    }

    try {
      setConnectionStatus('Connecting...');
      const response = await fetch(`${API_BASE}/api/lifeguards/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          name: nameInput.trim(), 
          phone: phoneInput.trim() 
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setLifeguardId(data.id);
        setLifeguardName(data.name);
        setAssignedZones(data.zones || []);
        await saveSession(data.id, data.name);
        setIsLoggedIn(true);
        setIsConnected(true);
        setConnectionStatus('Connected');
        Alert.alert('Welcome!', data.message || 'Registration successful');
      } else {
        Alert.alert('Error', data.error || 'Registration failed');
        setConnectionStatus('Disconnected');
      }
    } catch (error) {
      console.error('Login error:', error);
      Alert.alert(
        'Connection Error',
        `Cannot connect to server.\n\nMake sure:\n1. Backend is running on your PC\n2. API_BASE in App.js is set to your PC's IP address\n3. Your phone and PC are on the same WiFi\n\nCurrent API: ${API_BASE}`
      );
      setConnectionStatus('Disconnected');
    }
  };

  const handleLogout = async () => {
    Alert.alert('Logout', 'Are you sure you want to logout?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Logout',
        style: 'destructive',
        onPress: async () => {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
          }
          await clearSession();
          setIsLoggedIn(false);
          setLifeguardId(null);
          setLifeguardName('');
          setAlerts([]);
          setIsConnected(false);
          setConnectionStatus('Disconnected');
        },
      },
    ]);
  };

  const startPolling = () => {
    // Clear existing interval
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    // Initial load
    loadAlerts();

    // Poll every 3 seconds for new alerts
    pollIntervalRef.current = setInterval(() => {
      loadAlerts();
    }, 3000);
  };

  const loadAlerts = async () => {
    if (!lifeguardId) return;

    try {
      const response = await fetch(
        `${API_BASE}/api/lifeguards/${lifeguardId}/alerts?limit=30`
      );
      const data = await response.json();

      if (data.alerts) {
        // Check for new alerts
        const newAlerts = data.alerts.filter(
          alert => !alerts.find(a => a.id === alert.id || a.ts === alert.ts)
        );

        if (newAlerts.length > 0 && alerts.length > 0) {
          // Vibrate for new alerts
          Vibration.vibrate([0, 500, 200, 500]);
        }

        setAlerts(data.alerts);
        setAssignedZones(data.assigned_zones || []);
        setIsConnected(true);
        setConnectionStatus('Live');
      }
    } catch (error) {
      console.error('Error loading alerts:', error);
      setIsConnected(false);
      setConnectionStatus('Reconnecting...');
    }
  };

  const sendHeartbeat = async () => {
    if (!lifeguardId) return;

    try {
      await fetch(`${API_BASE}/api/lifeguards/${lifeguardId}/heartbeat`, {
        method: 'POST',
      });
    } catch (error) {
      // Silent fail for heartbeat
    }
  };

  const handleRespond = async (alert) => {
    try {
      Vibration.vibrate(100);
      
      await fetch(`${API_BASE}/api/lifeguards/${lifeguardId}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alert_id: alert.id,
          zone: alert.zone,
        }),
      });

      Alert.alert(
        '✓ Response Recorded',
        `You are responding to Zone ${alert.zone}`,
        [{ text: 'OK' }]
      );
    } catch (error) {
      console.error('Error responding:', error);
    }
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return 'Just now';
    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);

      if (diffMins < 1) return 'Just now';
      if (diffMins < 60) return `${diffMins}m ago`;

      const diffHours = Math.floor(diffMins / 60);
      if (diffHours < 24) return `${diffHours}h ago`;

      return date.toLocaleDateString();
    } catch {
      return 'Just now';
    }
  };

  // ============ RENDER LOGIN SCREEN ============
  if (!isLoggedIn) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#0d47a1" />
        
        <View style={styles.loginHeader}>
          <Text style={styles.logo}>🏖️</Text>
          <Text style={styles.title}>CoastVision</Text>
          <Text style={styles.subtitle}>Lifeguard Alert System</Text>
        </View>

        <View style={styles.loginCard}>
          <Text style={styles.loginTitle}>👋 Lifeguard Login</Text>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Your Name</Text>
            <TextInput
              style={styles.input}
              placeholder="Enter your name"
              placeholderTextColor="#aaa"
              value={nameInput}
              onChangeText={setNameInput}
              autoCapitalize="words"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Phone (Optional)</Text>
            <TextInput
              style={styles.input}
              placeholder="For quick contact"
              placeholderTextColor="#aaa"
              value={phoneInput}
              onChangeText={setPhoneInput}
              keyboardType="phone-pad"
            />
          </View>

          <TouchableOpacity style={styles.loginButton} onPress={handleLogin}>
            <Text style={styles.loginButtonText}>Start Monitoring</Text>
          </TouchableOpacity>

          <Text style={styles.serverInfo}>Server: {API_BASE}</Text>
          <Text style={styles.connectionStatusText}>{connectionStatus}</Text>
        </View>
      </SafeAreaView>
    );
  }

  // ============ RENDER DASHBOARD ============
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0d47a1" />

      {/* Connection Bar */}
      <View style={[styles.connectionBar, isConnected ? styles.connected : styles.disconnected]}>
        <View style={[styles.statusDot, isConnected && styles.statusDotPulse]} />
        <Text style={styles.connectionText}>{connectionStatus}</Text>
      </View>

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.userName}>{lifeguardName}</Text>
          <Text style={styles.userZones}>
            Monitoring: {assignedZones.length > 0 ? `Zone ${assignedZones.join(', ')}` : 'All Zones'}
          </Text>
        </View>
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      </View>

      {/* Alerts Section */}
      <Text style={styles.sectionTitle}>🚨 Recent Alerts</Text>

      <ScrollView style={styles.alertsList} contentContainerStyle={styles.alertsContent}>
        {alerts.length === 0 ? (
          <View style={styles.noAlerts}>
            <Text style={styles.noAlertsIcon}>✅</Text>
            <Text style={styles.noAlertsText}>All Clear</Text>
            <Text style={styles.noAlertsSubtext}>No active alerts</Text>
          </View>
        ) : (
          alerts.map((alert, index) => {
            const isAdmin = alert.type === 'admin_alert';
            const isDrowning = (alert.label || '').toLowerCase().includes('drown');

            return (
              <View
                key={alert.id || alert.ts || index}
                style={[
                  styles.alertCard,
                  isDrowning && styles.alertCardDrowning,
                  isAdmin && styles.alertCardAdmin,
                ]}
              >
                <View style={styles.alertHeader}>
                  <Text style={styles.alertZone}>
                    {isAdmin ? '📢 Admin Alert' : `Zone ${alert.zone}`}
                  </Text>
                  <Text style={styles.alertTime}>{formatTime(alert.ts || alert.timestamp)}</Text>
                </View>

                <View style={[styles.alertLabel, isDrowning ? styles.labelDrowning : styles.labelAdmin]}>
                  <Text style={styles.alertLabelText}>
                    {isAdmin ? 'BROADCAST' : (alert.label || 'ALERT').toUpperCase()}
                  </Text>
                </View>

                {alert.message && (
                  <Text style={styles.alertMessage}>{alert.message}</Text>
                )}

                {alert.conf && (
                  <Text style={styles.alertConfidence}>
                    Confidence: {(alert.conf * 100).toFixed(0)}%
                  </Text>
                )}

                <TouchableOpacity
                  style={styles.respondButton}
                  onPress={() => handleRespond(alert)}
                >
                  <Text style={styles.respondButtonText}>🏃 I'm Responding</Text>
                </TouchableOpacity>
              </View>
            );
          })
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

// ============ STYLES ============
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0d47a1',
  },

  // Login Screen
  loginHeader: {
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 40,
  },
  logo: {
    fontSize: 64,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 10,
  },
  subtitle: {
    fontSize: 16,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },
  loginCard: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 20,
    margin: 20,
    padding: 24,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  loginTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
    marginBottom: 24,
  },
  inputGroup: {
    marginBottom: 20,
  },
  label: {
    color: 'rgba(255,255,255,0.9)',
    fontSize: 14,
    marginBottom: 8,
  },
  input: {
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: 12,
    padding: 14,
    color: 'white',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  loginButton: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 10,
  },
  loginButtonText: {
    color: '#0d47a1',
    fontSize: 16,
    fontWeight: 'bold',
  },
  serverInfo: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 11,
    textAlign: 'center',
    marginTop: 16,
  },
  connectionStatusText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 4,
  },

  // Connection Bar
  connectionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
  },
  connected: {
    backgroundColor: '#4caf50',
  },
  disconnected: {
    backgroundColor: '#f44336',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'white',
    marginRight: 8,
  },
  statusDotPulse: {
    opacity: 0.8,
  },
  connectionText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },

  // Header
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: 'rgba(255,255,255,0.1)',
    marginHorizontal: 16,
    marginTop: 16,
    borderRadius: 16,
  },
  userName: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  userZones: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    marginTop: 2,
  },
  logoutButton: {
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  logoutText: {
    color: 'white',
    fontSize: 12,
  },

  // Alerts Section
  sectionTitle: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginTop: 24,
    marginHorizontal: 20,
    marginBottom: 12,
  },
  alertsList: {
    flex: 1,
    paddingHorizontal: 16,
  },
  alertsContent: {
    paddingBottom: 20,
  },

  // No Alerts
  noAlerts: {
    alignItems: 'center',
    padding: 40,
  },
  noAlertsIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  noAlertsText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  noAlertsSubtext: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 14,
    marginTop: 4,
  },

  // Alert Cards
  alertCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#4caf50',
  },
  alertCardDrowning: {
    borderLeftColor: '#f44336',
    backgroundColor: '#fff5f5',
  },
  alertCardAdmin: {
    borderLeftColor: '#ff9800',
    backgroundColor: '#fff8e1',
  },
  alertHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  alertZone: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#0d47a1',
  },
  alertTime: {
    fontSize: 11,
    color: '#666',
  },
  alertLabel: {
    alignSelf: 'flex-start',
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    marginBottom: 8,
  },
  labelDrowning: {
    backgroundColor: '#f44336',
  },
  labelAdmin: {
    backgroundColor: '#ff9800',
  },
  alertLabelText: {
    color: 'white',
    fontSize: 11,
    fontWeight: 'bold',
  },
  alertMessage: {
    color: '#666',
    fontSize: 13,
    marginBottom: 8,
  },
  alertConfidence: {
    color: '#666',
    fontSize: 12,
    marginBottom: 12,
  },
  respondButton: {
    backgroundColor: '#0d47a1',
    borderRadius: 10,
    padding: 12,
    alignItems: 'center',
  },
  respondButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
});
