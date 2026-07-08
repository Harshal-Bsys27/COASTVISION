# CoastVision Mobile Dev Client Workflow (Android)

This workflow replaces day-to-day dependence on Expo Go and keeps fast iteration.

## One-time setup

1. Connect Android phone to laptop via USB.
2. Enable Developer options + USB debugging on phone.
3. From repo root:

```powershell
Set-Location "C:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION"
.\run_mobile.ps1 -BuildDevClient
```

This builds and installs your own CoastVision dev app (`expo-dev-client`) on the phone.

## Daily development (after one-time install)

1. USB is NOT required for normal coding iteration.
2. Keep phone and laptop on same Wi-Fi/hotspot network.
3. Start backend on laptop.
4. Start mobile dev client bundler:

```powershell
Set-Location "C:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION"
.\run_mobile.ps1 -DevClient
```

5. Open the installed CoastVision app icon on Android.
6. App connects to Metro and supports Fast Refresh.

## When USB is required

USB is required only when:

- first installing dev client,
- rebuilding native Android app after native dependency/config changes,
- ADB/network debugging tasks.

USB is not required for normal JavaScript/React code changes.

## Optional fallback

If LAN is unstable for a session, you can still use tunnel mode:

```powershell
Set-Location "C:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION\frontend\mobile"
npm run start:devclient:tunnel
```

## Notes for local backend

- Use laptop LAN IP in the app backend URL, not `localhost`.
- Example: `http://<laptop-ip>:8000`
- Verify from phone browser first: `http://<laptop-ip>:8000/api/health`
