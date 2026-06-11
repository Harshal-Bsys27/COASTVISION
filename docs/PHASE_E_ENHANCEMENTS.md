# Phase E — Optional Enhancements

Post-demo polish for the lifeguard mobile app. Phases A–D are required; Phase E items are optional.

## Implemented

| Feature | Description |
| -------- | ------------- |
| **Heartbeat** | While signed in, mobile sends `POST /api/lifeguards/{id}/heartbeat` every 60s. Web admin **Lifeguard Accounts** tab shows online status. |
| **Foreground zone refresh** | When the app returns to foreground, `GET /api/lifeguards/me` refreshes assigned zones — no sign-out needed after admin changes zones on web. |
| **Respond on alerts** | **Logs** tab: **Respond** on drowning and high-crowd events → `POST /api/lifeguards/{id}/respond`. Response time appears in snackbar and **Analytics → Responses**. |

### Files added/changed

- `frontend/mobile/src/hooks/useLifeguardHeartbeat.js`
- `frontend/mobile/src/hooks/useRefreshOnForeground.js`
- `frontend/mobile/src/components/LifeguardSessionEffects.js`
- `frontend/mobile/src/components/ActivityEventCard.js` — Respond button
- `frontend/mobile/src/screens/EventLogsScreen.js` — respond handler
- `frontend/mobile/src/utils/activityEvents.js` — `respondable` + `alertId` on events

## Viva demo (Phase E)

1. **Web:** Lifeguards tab — confirm lifeguard shows **Online** after mobile sign-in (wait up to 60s or background/foreground app).
2. **Web:** Change Sara’s zones (e.g. `[1,3]` → `[2]`) → on phone, switch away and back to CoastVision → Dashboard updates without re-login.
3. **Mobile:** Logs → drowning or high-crowd event → tap **Respond** → snackbar shows response time → **Analytics → Responses** tab lists the entry.

## Not implemented (future)

| Feature | Notes |
| -------- | ------ |
| ~~Mobile SSE alert stream~~ | Implemented — `useLifeguardAlertStream` (XHR-based SSE) |
| FCM push notifications | Requires Firebase project + `google-services.json` |
| Admin PIN on web | Local PIN gate before dashboard |
| Server-side stream ACLs | Zone-scoped MJPEG/HLS by lifeguard token |
| ~~Remove Telegram backend~~ | Removed — `telegram_notify.py` and `/api/telegram/*` deleted |
| Standalone APK | See mobile README **Build APK** |

## Build APK (optional)

```powershell
cd frontend\mobile
npm install
npx expo prebuild
npx expo run:android
```

Requires Android SDK / USB debugging or emulator. Produces a dev build without Expo Go.
