"""Phase A API smoke test — run with backend already started."""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
failures = []


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=45) as r:
        return json.loads(r.read().decode()), r.status


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:
        print(f"FAIL  {name}: {e}")
        failures.append(name)


def main():
    def health():
        data, _ = get("/api/health")
        assert data.get("status") == "ok", data
        assert data.get("zones", 0) >= 1, "no zones running"
        print(f"       device={data.get('device')} zones={data.get('zones')}")

    def zones():
        data, _ = get("/api/zones")
        items = data.get("items") or []
        assert len(items) >= 1, "no zone items"
        active = [z for z in items if z.get("active")]
        assert active, "no active zones"
        print(f"       {len(items)} zones, {len(active)} active")

    def alerts():
        data, _ = get("/api/alerts?limit=5")
        assert "items" in data, data

    def analysis():
        data, _ = get("/api/analysis")
        assert "alerts_total" in data or "alerts_by_zone" in data, data

    def crowd():
        data, _ = get("/api/analytics/crowd-status")
        assert isinstance(data, dict), data

    def frame():
        zones_data, _ = get("/api/zones")
        zid = zones_data["items"][0]["id"]
        req = urllib.request.Request(f"{BASE}/api/zones/{zid}/frame.jpg")
        with urllib.request.urlopen(req, timeout=45) as r:
            body = r.read()
            assert len(body) > 1000, f"frame too small ({len(body)} bytes)"
            assert "image" in (r.headers.get("Content-Type") or ""), r.headers.get("Content-Type")

    check("A1 Backend health", health)
    check("A2 Zones API", zones)
    check("A3 Alerts API", alerts)
    check("A4 Analysis API", analysis)
    check("A5 Crowd analytics API", crowd)
    check("A6 Frame stream (frame.jpg)", frame)

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)

    print("\nAll Phase A API checks passed.")
    print("Manual steps remaining:")
    print("  - Mobile: npm start in frontend/mobile, set laptop IP in Settings")
    print("  - Web: .\\run_frontend.ps1, open http://localhost:5173")


if __name__ == "__main__":
    main()
