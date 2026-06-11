# Phase A — Stabilization Checklist

Goal: reliable end-to-end demo on laptop + Wi-Fi before lifeguard auth (Phase B).

## Prerequisites

| Item | Location | Status |
|------|----------|--------|
| Python venv | `venv\Scripts\python.exe` | Required |
| Model weights | `yolov8n.pt` or `models\yolov8n.pt` | Required |
| Zone videos | `frontend\dashboard\videos\zone1.mp4` … | Required |
| Node.js 18+ | For web + mobile | Required |

## Quick Start (3 terminals)

```powershell
# Terminal 1 — backend
cd "c:\Users\Shalini\Downloads\coastvision app"
.\run_backend.ps1

# Terminal 2 — web admin dashboard
.\run_frontend.ps1
# Open http://localhost:5173

# Terminal 3 — mobile (Expo Go)
.\run_mobile.ps1
# Scan QR → Settings → http://<laptop-ip>:8000 → Test Connection → Save
```

Find laptop IP: `ipconfig` → look for IPv4 on your Wi-Fi adapter (e.g. `192.168.1.5`).

## Automated Smoke Test

With backend running:

```powershell
.\scripts\phase_a_smoke_test.ps1
```

Checks: health, zones, alerts, analysis, crowd analytics, frame.jpg stream.

## Phase A Exit Criteria

| # | Check | How to verify |
|---|-------|---------------|
| A1 | Backend starts | `.\run_backend.ps1` — no import/model errors |
| A2 | Health OK | `GET /api/health` → `status: ok`, `zones >= 1` |
| A3 | Zones active | `GET /api/zones` → all zones `active: true`, `is_opened: true` |
| A4 | Frame streams | `GET /api/zones/1/frame.jpg` returns JPEG (>1 KB) |
| A5 | Alerts load | `GET /api/alerts?limit=5` → `items` array |
| A6 | Analytics load | `GET /api/analysis` → chart data keys present |
| A7 | Web builds | `cd frontend\web && npm run build` succeeds |
| A8 | Web tabs load | Dashboard, Analytics, Logs, Lifeguards, Videos, Settings |
| A9 | Mobile connects | Settings → Test Connection → Connected, zone count shown |
| A10 | Mobile dashboard | Zone cards show live frame.jpg previews |
| A11 | Mobile logs | Event Logs tab shows alert history |
| A12 | Same Wi-Fi | Phone uses `http://<laptop-ip>:8000`, not `127.0.0.1` |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `venv not found` | `python -m venv venv` then `pip install -r requirements.txt` |
| `No model weights found` | Place `yolov8n.pt` in project root or `models/` |
| No zones / inactive | Add `zone1.mp4` … `zone5.mp4` to `frontend\dashboard\videos\` |
| Mobile network failed | Same Wi-Fi; Windows Firewall allow port 8000 inbound |
| Use laptop IP on phone | `127.0.0.1` only works on the laptop itself |
| CPU mode (no GPU) | Normal — `run_backend.ps1` auto-selects `cpu` when CUDA unavailable |
| Expo Go crash on video | App uses frame.jpg polling only (no expo-av) |

## Verified Baseline (Phase A complete)

Run date: 2026-06-06

- Backend: `status=ok`, 5 zones, device=cpu
- All 5 zone videos active and streaming frames
- Alerts, analysis, crowd-status APIs responding
- Web frontend production build succeeds
- Mobile app structure ready (Expo Go, frame.jpg streams, 5 tabs)

**Next:** Phase B — lifeguard phone login (`SignInScreen` + backend `/login` + `/me`).
