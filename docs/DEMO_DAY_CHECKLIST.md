# CoastVision — Demo Day Checklist

One-page runbook for viva / presentation day.

## Before you leave home

- [ ] Laptop charged; phone charged
- [ ] Phone and laptop on **same Wi-Fi** (no guest network isolation)
- [ ] Expo Go updated (Play Store)
- [ ] Run smoke tests once:
  ```powershell
  cd "c:\Users\Shalini\Downloads\coastvision app"
  .\run_backend.ps1
  # New terminal:
  .\scripts\phase_a_smoke_test.ps1
  .\scripts\phase_b_e_smoke_test.ps1
  ```

## Three terminals (in order)

### Terminal 1 — Backend

```powershell
cd "c:\Users\Shalini\Downloads\coastvision app"
.\run_backend.ps1
```

Wait for: `Running on http://0.0.0.0:8000` and zones active.

### Terminal 2 — Web admin

```powershell
cd "c:\Users\Shalini\Downloads\coastvision app"
.\run_frontend.ps1
```

Open: **http://localhost:5173**

### Terminal 3 — Mobile

```powershell
cd "c:\Users\Shalini\Downloads\coastvision app\frontend\mobile"
npm start
```

Scan QR in **Expo Go** (not a dev client).

## Network

1. Find laptop IP: `ipconfig` → IPv4 (e.g. `192.168.1.4`)
2. On phone Sign In screen: `http://<laptop-ip>:8000`
3. Windows Firewall: allow **port 8000** (and 8081 for Metro if needed)

## Pre-seeded test accounts

| Name | Phone | Zones | Role in demo |
|------|-------|-------|----------------|
| Raj Test | `9876543210` | Zone 2 | Single-zone lifeguard |
| Sara | `7718085148` | Zones 1, 3 | Multi-zone lifeguard |

Accounts live in [`data/alerts/lifeguards.json`](../data/alerts/lifeguards.json). Admin can change zones on web **Lifeguard Accounts** tab.

## Demo flow (5 min)

1. **Web:** Dashboard — all zones visible
2. **Web:** Lifeguard Accounts — show Raj / Sara, zone chips, **Online** status
3. **Mobile:** Sign in as Sara → Dashboard shows zones **1 and 3 only**
4. **Mobile:** Logs → **Respond** on alert → Analytics → Responses
5. **Talking point:** Same backend, two clients — admin unscoped, lifeguard scoped

## If something breaks

| Problem | Fix |
|---------|-----|
| Mobile "Network request failed" | Wrong IP; use `http://` not `https://`; same Wi-Fi |
| Two backends on port 8000 | `.\run_backend.ps1 -Stop` then restart |
| Expo Go closed / no bundle | `cd frontend\mobile && npm start`, press `r` to reload |
| Sara sees all zones | Sign out/in, or background app and reopen (foreground refresh) |
| crowd-alerts 500 | `.\run_backend.ps1 -Stop` then restart backend |
| Web blank | Check Terminal 2; open http://localhost:5173 |
| No drowning events in Logs | Normal if no detections — use crowd events or explain polling |

## Optional: standalone APK

If Expo Go fails on demo day, build ahead of time:

```powershell
.\scripts\build_mobile_apk.ps1
```

See [`frontend/mobile/README.md`](../frontend/mobile/README.md). Native `android/` folder is created by `expo prebuild`.

## After demo

```powershell
.\run_backend.ps1 -Stop
```

Ctrl+C in web and mobile terminals.
