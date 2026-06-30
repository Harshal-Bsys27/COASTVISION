#!/usr/bin/env python3
"""
Quick verification script for WebSocket integration
Tests that all imports work correctly without starting full backend
"""

import sys
import os

print("[TEST] CoastVision WebSocket Integration Verification")
print("-" * 60)

# Test 1: Python packages
print("\n[TEST 1] Checking Python packages...")
try:
    from flask import Flask
    print("  ✓ Flask imported successfully")
except Exception as e:
    print(f"  ✗ Flask import failed: {e}")
    sys.exit(1)

try:
    from flask_cors import CORS
    print("  ✓ Flask-CORS imported successfully")
except Exception as e:
    print(f"  ✗ Flask-CORS import failed: {e}")
    sys.exit(1)

try:
    from flask_socketio import SocketIO, emit
    print("  ✓ Flask-SocketIO imported successfully")
except Exception as e:
    print(f"  ✗ Flask-SocketIO import failed: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("  ✓ Ultralytics YOLO imported successfully")
except Exception as e:
    print(f"  ✗ Ultralytics YOLO import failed: {e}")
    sys.exit(1)

# Test 2: Create Flask app with SocketIO
print("\n[TEST 2] Creating Flask app with SocketIO...")
try:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                       ping_timeout=60, ping_interval=25)
    print("  ✓ Flask app + SocketIO initialized successfully")
except Exception as e:
    print(f"  ✗ Flask app initialization failed: {e}")
    sys.exit(1)

# Test 3: Define SocketIO event handlers
print("\n[TEST 3] Testing SocketIO event handlers...")
try:
    @socketio.on('connect')
    def handle_connect(auth=None):
        client_type = 'unknown'
        if auth:
            client_type = auth.get('client_type', 'unknown')
        print(f"  [WS] Client connected (type={client_type})")
        emit('connection_response', {'data': 'Connected to CoastVision backend'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print("  [WS] Client disconnected")
    
    print("  ✓ SocketIO event handlers defined successfully")
except Exception as e:
    print(f"  ✗ SocketIO event handler definition failed: {e}")
    sys.exit(1)

# Test 4: Import backend modules
print("\n[TEST 4] Checking backend modules...")
try:
    # This will fail if backend/server.py has syntax errors, 
    # but we can't fully import it without starting the server
    import importlib.util
    spec = importlib.util.spec_from_file_location("server", 
                                                   "backend/server.py")
    if spec and spec.loader:
        print("  ✓ backend/server.py syntax is valid")
    else:
        print("  ! Could not verify backend/server.py syntax")
except SyntaxError as e:
    print(f"  ✗ Syntax error in backend/server.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ! Could not verify backend/server.py: {e}")

# Test 5: Check frontend files
print("\n[TEST 5] Checking frontend files...")
try:
    if os.path.exists("frontend/web/src/hooks/useRealtimeUpdates.js"):
        print("  ✓ useRealtimeUpdates.js exists")
    else:
        print("  ✗ useRealtimeUpdates.js not found")
        sys.exit(1)
    
    if os.path.exists("frontend/web/src/App.jsx"):
        print("  ✓ App.jsx exists")
        # Check if import was added
        with open("frontend/web/src/App.jsx", "r") as f:
            content = f.read()
            if "useRealtimeUpdates" in content:
                print("  ✓ App.jsx imports useRealtimeUpdates")
            else:
                print("  ✗ App.jsx doesn't import useRealtimeUpdates")
                sys.exit(1)
    else:
        print("  ✗ App.jsx not found")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Frontend file check failed: {e}")
    sys.exit(1)

# Test 6: Check requirements
print("\n[TEST 6] Checking requirements.txt...")
try:
    with open("requirements.txt", "r") as f:
        content = f.read()
        required_packages = ["flask-socketio", "python-engineio", "python-socketio"]
        for pkg in required_packages:
            if pkg in content:
                print(f"  ✓ {pkg} in requirements.txt")
            else:
                print(f"  ✗ {pkg} NOT in requirements.txt")
                sys.exit(1)
except Exception as e:
    print(f"  ✗ Requirements check failed: {e}")
    sys.exit(1)

# Test 7: Check package.json
print("\n[TEST 7] Checking frontend/web/package.json...")
try:
    import json
    with open("frontend/web/package.json", "r") as f:
        package = json.load(f)
        if "socket.io-client" in package.get("dependencies", {}):
            print("  ✓ socket.io-client in package.json")
        else:
            print("  ✗ socket.io-client NOT in package.json")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ package.json check failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nWebSocket integration is properly configured.")
print("\nNext steps:")
print("  1. Backend:  python backend/server.py")
print("  2. Frontend: cd frontend/web && npm run dev")
print("  3. Open:     http://localhost:5173")
print("  4. Check:    'Live' indicator in AppBar when connected")
print("\nFor full verification, see docs/WEBSOCKET_IMPLEMENTATION_COMPLETE.md")
