# CoastVision Lifeguard App

React Native mobile app for lifeguards to receive real-time drowning alerts.

## Quick Start (Easiest Method - Expo Go)

### Step 1: Install Expo Go on your phone
- **Android**: Download "Expo Go" from Play Store
- **iPhone**: Download "Expo Go" from App Store

### Step 2: Start the app
```powershell
cd LifeguardApp
npx expo start
```

### Step 3: Connect
- A QR code will appear in the terminal
- Open Expo Go app on your phone
- Scan the QR code
- The app will load on your phone!

**Important**: Your phone and PC must be on the same WiFi network.

---

## Build APK (Install without Expo Go)

### Option A: Build locally (requires Android SDK)

1. Install EAS CLI:
```powershell
npm install -g eas-cli
```

2. Build APK:
```powershell
cd LifeguardApp
npx expo prebuild --platform android
cd android
./gradlew assembleRelease
```

The APK will be at: `android/app/build/outputs/apk/release/app-release.apk`

### Option B: Build in cloud (Free - Expo account required)

1. Create free Expo account: https://expo.dev/signup

2. Login:
```powershell
npx eas login
```

3. Build APK:
```powershell
cd LifeguardApp
npx eas build --platform android --profile preview
```

4. Download the APK from your Expo dashboard

---

## Configuration

### Set your PC's IP Address

Edit `App.js` line 19:
```javascript
const API_BASE = 'http://YOUR_PC_IP:8000';
```

To find your PC's IP:
```powershell
ipconfig
```
Look for "IPv4 Address" under your WiFi adapter.

Current setting: `http://10.202.83.183:8000`

### Backend Setup

Make sure the backend is running on your PC:
```powershell
cd COASTVISION
.\run_backend.ps1
```

The backend must be accessible from your phone (same network).

---

## Features

- **Login**: Register as a lifeguard
- **Real-time alerts**: Receive drowning detection alerts
- **Zone assignment**: Admin can assign you to specific zones
- **Response tracking**: Press "I'm Responding" to notify control room
- **Vibration alerts**: Phone vibrates when new alerts arrive

---

## Troubleshooting

### "Cannot connect to server"
1. Check backend is running: `http://YOUR_PC_IP:8000/api/health`
2. Ensure phone and PC are on same WiFi
3. Try disabling Windows Firewall temporarily
4. Check API_BASE IP in App.js matches your PC

### App won't load on Expo Go
1. Make sure you're on same WiFi network
2. Try pressing 's' in terminal to switch to tunnel mode
3. Check if any VPN is blocking connection

### QR code not scanning
- Press 'c' in terminal to show QR code again
- Try typing the URL shown below QR code manually in Expo Go

---

## Development

```powershell
cd LifeguardApp
npm start          # Start Expo dev server
npm run android    # Run on connected Android device
npm run ios        # Run on iOS simulator (Mac only)
```
