from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app)

alerts = []
ALERT_IMAGES_DIR = "alert_images"

if not os.path.exists(ALERT_IMAGES_DIR):
    os.makedirs(ALERT_IMAGES_DIR)

@app.route('/alert', methods=['POST'])
def receive_alert():
    """Receive AI alert + optional image"""
    data = request.json
    alert_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_filename = None
    if 'image' in data:
        img_bytes = base64.b64decode(data['image'])
        image_filename = f"{alert_time}.jpg"
        with open(os.path.join(ALERT_IMAGES_DIR, image_filename), "wb") as f:
            f.write(img_bytes)

    alert = {
        'id': len(alerts),
        'timestamp': alert_time,
        'zone': data['zone'],
        'status': data['status'],
        'image_url': f"/alert_image/{image_filename}" if image_filename else None,
        'acknowledged': False
    }
    alerts.append(alert)

    return jsonify({"status": "success", "alert_id": alert['id']})

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Fetch alerts after a given ID (for polling in Flutter)"""
    last_id = request.args.get('last_id', default=-1, type=int)
    new_alerts = [a for a in alerts if a['id'] > last_id]
    return jsonify(new_alerts)

@app.route('/alert_image/<path:image_path>')
def get_alert_image(image_path):
    """Serve saved alert images"""
    return send_file(os.path.join(ALERT_IMAGES_DIR, image_path))

@app.route('/acknowledge_alert', methods=['POST'])
def acknowledge_alert():
    """Lifeguard acknowledges an alert"""
    data = request.json
    alert_id = data.get('alert_id')

    for alert in alerts:
        if alert['id'] == alert_id:
            alert['acknowledged'] = True
            return jsonify({"status": "acknowledged", "alert_id": alert_id})

    return jsonify({"error": "Alert not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
