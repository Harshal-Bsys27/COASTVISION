import requests

def send_emergency_alert(zone, timestamp, image_path):
    # Example: Send POST request to Flutter app backend
    url = "http://your-flutter-backend/api/alert"
    data = {
        "zone": zone,
        "timestamp": timestamp,
        "image_path": image_path
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"✅ Notification sent to lifeguard app for Zone {zone}")
        else:
            print(f"❌ Failed to notify lifeguard app: {response.text}")
    except Exception as e:
        print(f"❌ Error sending notification: {e}")
