# CoastVision Mobile (React Native)

Android app for CoastVision with Dashboard, Analytics, Event Logs, and Settings. Lifeguard accounts are managed on the web admin dashboard.

## Prerequisites

- Node.js 18+
- Expo Go on your Android phone (Play Store)
- CoastVision backend running on your laptop (`.\run_backend.ps1`)
- Phone and laptop on the **same Wi-Fi**

## Phase A (stabilization)

Before lifeguard login (Phase B), confirm the full stack works:

```powershell
.\run_backend.ps1
.\scripts\phase_a_smoke_test.ps1
```

See [docs/PHASE_A_STABILIZATION.md](../../docs/PHASE_A_STABILIZATION.md) for the full checklist.

## Quick Start

```powershell
# Terminal 1 — backend
cd "c:\Users\Shalini\Downloads\coastvision app"
.\run_backend.ps1

# Terminal 2 — find laptop IP
ipconfig

# Terminal 3 — mobile app
cd frontend\mobile
npm install
npm start
```

Use `npm start` — it runs `expo start --go` (Expo Go mode).

1. Scan the QR code with Expo Go.
2. On the **Sign In** screen, enter `http://<your-laptop-ip>:8000` and your phone number.
3. Tap **Sign In** (admin must create your account first — see below).

### Create a lifeguard account (admin)

On the **web dashboard** → **Lifeguards** tab → enter name + phone → **Create Account** → assign zones with the zone chips.

Then sign in on mobile with that phone number.

## Phase E (enhancements)

- **Heartbeat** — keeps lifeguard **Online** on web admin (every 60s while app is open).
- **Zone refresh** — return to the app after admin changes your zones on web; no sign-out needed.
- **Respond** — on **Logs**, tap **Respond** on drowning / high-crowd events; see response time in **Analytics → Responses**.

See [docs/PHASE_E_ENHANCEMENTS.md](../../docs/PHASE_E_ENHANCEMENTS.md).

## Project Structure

```
mobile/src/
  screens/       SignIn, Dashboard, Analytics, EventLogs, Settings, ZoneDetail
  components/    ZoneCard, ActivityEventCard, ZoneStream, LifeguardSessionEffects
  navigation/    Bottom tabs + dashboard stack
  hooks/         usePollApi, useAlertNotifications, useLifeguardHeartbeat
  context/       ApiContext (auth, server URL, API client)
```

Shared API client: `frontend/shared/api.js`

## Build APK (optional)

From project root (requires Android SDK):

```powershell
.\scripts\build_mobile_apk.ps1
```

Or manually:

```powershell
cd frontend\mobile
npm run prebuild
npm run build:android
```

Prebuild generates `frontend/mobile/android/`. A full release APK needs Android Studio or `npx expo run:android --variant release` with SDK installed.

## Seeing logs in the terminal (Expo Go)

After the app loads on your phone, logs appear in the **same terminal** where Metro is running.

1. Start with: `npm start` (forces Expo Go; avoid "development build" mode)
2. Scan QR in **Expo Go** app (not a dev client)
3. Open the app on your phone — you should see:
   ```
   LOG  [CoastVision] App started — logs appear in this terminal when using Expo Go
   ```
4. In the Metro terminal, press **`j`** to open the JavaScript debugger (optional)
5. If still no logs: shake phone → **Reload**, or press **`r`** in the terminal

**Common reasons logs are missing:**

| Cause | Fix |
|-------|-----|
| Switched to "development build" | Press **`s`** in terminal to switch back to Expo Go, or restart with `npm start` |
| App not connected to Metro | Same Wi-Fi; rescan QR code |
| Only "Android Bundled" shown | Normal until app runs — open app on phone after bundle completes |
| API errors | Now logged as `[CoastVision] zones poll failed` etc. |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Network request failed | Same Wi-Fi, correct IP, Windows Firewall allows port 8000 |
| Video not playing | App uses frame.jpg polling (Expo Go safe). Ensure backend is running and zones exist |
| `ExponentAV` / runtime not ready | Fixed — app no longer uses expo-av. Restart with `npm start` and reload app |
| Expo Go version mismatch | Update **Expo Go** from Play Store to latest version |
| Connection works in browser but not app | Use `http://` not `https://` for local Wi-Fi |
