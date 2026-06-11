# 🚀 Integration Guide: WebSocket + Dashboard + Model Training

Complete step-by-step guide to integrate real-time WebSocket updates with the dashboard and train unified YOLOv11 model.

---

## **PART 1: Backend WebSocket Integration** (1-2 hours)

### Step 1.1: Install Flask-SocketIO

```bash
pip install python-socketio python-engineio python-dotenv
pip install --upgrade ultralytics  # For YOLOv11
```

### Step 1.2: Update `backend/server.py`

Add imports at the top (line ~45):

```python
# After existing imports, add:
from websocket_integration import init_socketio, broadcast_alert, broadcast_lifeguard_response
```

Initialize SocketIO after Flask app creation (line ~1354):

```python
# Current code
app = Flask(__name__)
CORS(app)

# ADD THIS:
socketio = init_socketio(app, cors_allowed_origins="*")
```

### Step 1.3: Broadcast Alerts When Detected

Find `_persist_alerts()` function (line ~1128) and add broadcast call:

```python
def _persist_alerts(zid: int, alerts, annotated_bgr_frame):
    """Persist alerts to CSV and disk."""
    
    # ... existing code ...
    
    for alert_dict in alerts:
        # NEW: Broadcast to all clients (dashboard + mobile)
        broadcast_alert({
            'alert_id': alert_dict.get('alert_id'),
            'zone': _get_zone_display_name(zid),
            'type': alert_dict.get('type'),
            'confidence': alert_dict.get('confidence'),
            'timestamp': alert_dict.get('timestamp'),
            'image_url': f"/api/alerts/{alert_dict.get('alert_id')}/image.jpg",
        })
    
    # ... rest of existing code ...
```

### Step 1.4: Broadcast Lifeguard Responses

Find `/api/telegram/<lg_id>/pause` endpoint (line ~1850-1900) and add:

```python
@app.route("/api/telegram/<lg_id>/pause", methods=["POST"])
def pause_lifeguard(lg_id):
    """Pause lifeguard notifications (NEW: with WebSocket broadcast)"""
    
    notifier.set_paused(lg_id, True)
    
    # NEW: Broadcast to all clients
    broadcast_lifeguard_response(
        alert_id='system',
        lifeguard_id=lg_id,
        response_status='paused',
    )
    
    return jsonify({"status": "paused"})
```

### Step 1.5: Run Backend with WebSocket

```bash
cd backend
python server.py
# Should see: [init] WebSocket initialized
```

---

## **PART 2: Frontend Dashboard Integration** (1-2 hours)

### Step 2.1: Install Socket.IO Client

```bash
cd frontend/web
npm install socket.io-client
```

### Step 2.2: Update `App.jsx` to Use WebSocket

Add at the top of the file (line ~1):

```javascript
import useRealtimeUpdates from './hooks/useRealtimeUpdates';
```

Add in App component (inside main App function, after state declarations):

```javascript
export default function App() {
  // Existing state...
  const [alerts, setAlerts] = useState([]);
  const [zones, setZones] = useState([]);
  
  // NEW: Real-time WebSocket hook
  const { isConnected } = useRealtimeUpdates(
    // onNewAlert
    (alert) => {
      setAlerts(prev => [alert, ...prev.slice(0, 99)]);
      // Flash notification
      console.log('🚨 Alert received:', alert);
    },
    // onZoneUpdate
    (zoneData) => {
      setZones(prev => 
        prev.map(z => z.id === zoneData.zone_id ? { ...z, ...zoneData } : z)
      );
    },
    // onLifeguardResponse
    (response) => {
      console.log('✅ Response:', response);
      // Update response time in event logs
      setAlerts(prev => 
        prev.map(a => a.alert_id === response.alert_id 
          ? { ...a, response_time: response.response_time, responded_by: response.lifeguard_id }
          : a
        )
      );
    },
    // onSystemStatus
    (status) => {
      // Update system health indicators
      console.log('System:', status);
    }
  );
  
  // Add connection indicator to AppBar
  return (
    <div>
      <AppBar>
        <Toolbar>
          {/* Existing content... */}
          
          {/* Add WebSocket indicator */}
          <Box sx={{ ml: 'auto' }}>
            <Chip
              icon={<FiberManualRecordIcon />}
              label={isConnected ? 'Live' : 'Offline'}
              color={isConnected ? 'success' : 'error'}
              size="small"
            />
          </Box>
        </Toolbar>
      </AppBar>
      
      {/* Rest of dashboard... */}
    </div>
  );
}
```

### Step 2.3: Update Event Logs Table to Show Response Times

Find the "Event Logs" tab (line ~2200-2300) and update table to show response time:

```javascript
// In Event Logs table columns, add:
{
  field: 'response_time',
  headerName: 'Response Time',
  width: 120,
  renderCell: (params) => {
    const time = params.row.response_time;
    return time ? (
      <Chip 
        label={`${time.toFixed(1)}s`} 
        color={time < 60 ? 'success' : time < 120 ? 'warning' : 'error'}
        variant="outlined"
      />
    ) : (
      <Typography variant="caption" color="textSecondary">—</Typography>
    );
  },
}
```

### Step 2.4: Add Lifeguard Status Indicator

In Lifeguards tab, show connection status:

```javascript
<Chip
  label={isConnected ? '🟢 Connected' : '⚪ Offline'}
  color={isConnected ? 'success' : 'default'}
/>
```

---

## **PART 3: Train Unified YOLOv11 Model** (2-3 days data collection + 2-3 hours training)

### Step 3.1: Expand Dataset (See `docs/DATASET_EXPANSION.md` for details)

Minimum viable: Add 1,000+ "Person_Out" images

```bash
# Extract frames from video
ffmpeg -i beach_video.mp4 -vf fps=1 "frames/frame_%04d.jpg"

# Use Roboflow to annotate & augment
# Or use CVAT for manual annotation
```

### Step 3.2: Run Training Script

```bash
cd scripts
python train_unified_yolo11.py \
  --data ../dataset/data.yaml \
  --model n \
  --epochs 100 \
  --batch 16 \
  --device 0
```

**Expected output:**
```
🚀 CoastVision: Unified YOLOv11 Training
✅ GPU: NVIDIA GeForce RTX 3050 (6.0GB)
📦 Loading model: yolo11n.pt
⚖️  Class Weighting:
  - Drowning: 2.0x
  - Swimming: 1.0x
  - Person_Out: 4.0x ⭐

Training: 100% |████████████| 100/100 [2:45:30<00:00, ...]
✅ Training Complete!

🎯 Best Metrics:
  - mAP50: 0.876
  - mAP50-95: 0.592
```

### Step 3.3: Deploy New Model

```bash
# Copy best model to production location
cp runs/train/unified_yolo11/weights/best.pt models/best.pt

# Backend will auto-load it on restart
python backend/server.py
```

---

## **PART 4: Update YOLOv11 Priority in Backend** (Optional, 10 min)

Edit `backend/server.py` line ~588-605 to prioritize YOLOv11:

**BEFORE:**
```python
def _pick_model_path() -> Path:
    candidates = [
        ROOT / ".." / "models" / "best.pt",
        ROOT / ".." / "yolov8n.pt",      # YOLOv8 as fallback
        ROOT / ".." / "yolo11n.pt",      # YOLOv11 last
    ]
```

**AFTER:**
```python
def _pick_model_path() -> Path:
    candidates = [
        ROOT / ".." / "models" / "best.pt",
        ROOT / ".." / "yolo11n.pt",      # YOLOv11 first (better)
        ROOT / ".." / "yolov8n.pt",      # YOLOv8 fallback
    ]
```

---

## **PART 5: Verify Everything Works** (30 min)

### 5.1: Test Backend WebSocket

```bash
# Terminal 1: Start backend
cd backend
python server.py

# Terminal 2: Test WebSocket connection
python -c "
import socketio
import time

client = socketio.Client()
@client.on('new_alert')
def on_alert(data):
    print(f'✅ Alert received: {data}')

@client.on('pong')
def on_pong():
    print('✅ WebSocket working!')

client.connect('http://127.0.0.1:8000')
client.emit('ping')
time.sleep(2)
client.disconnect()
"
```

### 5.2: Test Dashboard

```bash
# Terminal 2: Start frontend
cd frontend/web
npm run dev

# Visit http://localhost:5173
# Should see "Live" status in top-right corner
```

### 5.3: Trigger a Test Alert

```bash
# Create test alert via REST API
curl -X POST http://127.0.0.1:8000/api/test-alert \
  -H "Content-Type: application/json" \
  -d '{"zone": 1, "type": "drowning"}'

# Should see alert popup on dashboard immediately
```

---

## **PART 6: Deployment** (1-2 hours)

### 6.1: Update `requirements.txt`

```bash
# Backend requirements
cd backend
pip install -r requirements.txt  # Should already have these now

# If missing, add:
echo "python-socketio==5.9.0" >> requirements.txt
echo "python-engineio==4.8.0" >> requirements.txt
```

### 6.2: Run with Production Server

```bash
# Use Waitress (production WSGI server)
pip install waitress

# Start with WebSocket support
python -c "
from backend.server import app, socketio
socketio.run(app, host='0.0.0.0', port=8000, debug=False)
"
```

### 6.3: Docker Deployment (Optional)

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY backend/ /app/backend/
COPY models/ /app/models/
RUN pip install -r backend/requirements.txt
CMD ["python", "-c", "from backend.server import app, socketio; socketio.run(app, host='0.0.0.0', port=8000)"]
```

```bash
docker build -t coastvision-backend .
docker run -p 8000:8000 coastvision-backend
```

---

## **TESTING CHECKLIST**

- [ ] Backend starts with `[init] WebSocket initialized` message
- [ ] Frontend connects: shows "Live" indicator
- [ ] Dashboard loads without errors
- [ ] Test alert triggers: appears on dashboard in < 500ms
- [ ] Lifeguard pause/resume works
- [ ] New model (`best.pt`) loads correctly
- [ ] Inference speed: 30-40ms per frame
- [ ] YOLOv11 model accuracy > 85% mAP50

---

## **TROUBLESHOOTING**

### Issue: WebSocket Connection Failed
```
Error: Connection refused

Solution:
1. Check backend is running: ps aux | grep server.py
2. Check port 8000 is open: netstat -an | grep 8000
3. Update CORS in init_socketio() if running on different domain
```

### Issue: Dashboard Blank After WebSocket
```
Error: TypeError: socket is undefined

Solution:
1. Check useRealtimeUpdates hook imported correctly
2. Verify socket.io-client installed: npm list socket.io-client
3. Check browser console for errors (F12)
```

### Issue: Model Not Loading
```
Error: RuntimeError: No model weights found

Solution:
1. Check models/best.pt exists
2. Check yolo11n.pt in root folder
3. Run: ls -la models/ && ls -la ../*.pt
```

---

## **PERFORMANCE EXPECTATIONS**

| Metric | Before | After |
|--------|--------|-------|
| **Inference** | 40-50ms | 30-40ms |
| **Accuracy (mAP50)** | 0.75 | 0.88+ |
| **Person_Out Detection** | 0.45 | 0.85+ |
| **Dashboard Latency** | 2-5s (polling) | <500ms (WebSocket) |
| **Mobile Alert Latency** | 3-5s | 2-5s (FCM) |

---

## **NEXT STEPS**

1. **Today**: Integrate WebSocket backend + update dashboard
2. **This Week**: Test and validate everything works
3. **Next Week**: Start dataset expansion (Person_Out class)
4. **Week 3**: Train unified YOLOv11 model
5. **Week 4**: Deploy new model to production

---

## **FILES CREATED/MODIFIED**

✅ Created:
- `backend/websocket_integration.py` - WebSocket event handlers
- `frontend/web/src/hooks/useRealtimeUpdates.js` - React hook
- `scripts/train_unified_yolo11.py` - Training script with class weighting
- `docs/DATASET_EXPANSION.md` - Dataset strategy guide
- `docs/INTEGRATION_GUIDE.md` - This file

📝 To Modify:
- `backend/server.py` - Add WebSocket imports & initialization
- `frontend/web/src/App.jsx` - Use useRealtimeUpdates hook
- `requirements.txt` - Add python-socketio, python-engineio

---

## **SUPPORT**

For questions, reach out to:
- **Harshal** (Architecture, Model Training)
- **Komal** (Frontend, WebSocket React)
- **Hardik** (Backend APIs)
- **Sara** (Data, DevOps)
