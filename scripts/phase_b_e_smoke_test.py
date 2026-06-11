"""Phase B–E API smoke test — lifeguard auth, zones, heartbeat, respond."""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
failures = []
TEST_PHONE = "9998887776"
TEST_NAME = "Smoke Test LG"


def request(method, path, body=None, headers=None):
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(BASE + path, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}, r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"error": raw}
        return payload, e.code


def get(path, headers=None):
    return request("GET", path, headers=headers)


def post(path, body=None, headers=None):
    return request("POST", path, body=body, headers=headers)


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:
        print(f"FAIL  {name}: {e}")
        failures.append(name)


def main():
    state = {"lg_id": None, "token": None, "zone_id": None}

    def register_and_login():
        data, status = post("/api/lifeguards/register", {"name": TEST_NAME, "phone": TEST_PHONE})
        assert status in (200, 201), f"register status {status}: {data}"
        state["lg_id"] = data.get("id")
        assert state["lg_id"], data

        data, status = post("/api/lifeguards/login", {"phone": TEST_PHONE})
        assert status == 200, f"login status {status}: {data}"
        state["token"] = data.get("session_token")
        assert state["token"], data
        print(f"       lg_id={state['lg_id']}")

    def me():
        data, status = get("/api/lifeguards/me", headers={"Authorization": f"Bearer {state['token']}"})
        assert status == 200, f"me status {status}: {data}"
        assert data.get("phone") == TEST_PHONE, data

    def assign_zone():
        zones_data, status = get("/api/zones")
        assert status == 200, zones_data
        items = zones_data.get("items") or []
        assert items, "no zones"
        state["zone_id"] = items[0]["id"]

        data, status = post(
            f"/api/lifeguards/{state['lg_id']}/assign",
            {"zones": [state["zone_id"]]},
        )
        assert status == 200, f"assign status {status}: {data}"
        assert state["zone_id"] in (data.get("zones") or []), data
        print(f"       assigned zone={state['zone_id']}")

    def scoped_alerts():
        data, status = get(f"/api/lifeguards/{state['lg_id']}/alerts?limit=20")
        assert status == 200, f"alerts status {status}: {data}"
        assigned = data.get("assigned_zones") or []
        assert assigned == [state["zone_id"]], assigned
        for alert in data.get("alerts") or []:
            zone = alert.get("zone")
            if zone is not None:
                assert zone in assigned or not assigned, alert
        print(f"       alerts={data.get('count', 0)}")

    def heartbeat():
        data, status = post(f"/api/lifeguards/{state['lg_id']}/heartbeat")
        assert status == 200, f"heartbeat status {status}: {data}"
        assert data.get("status") == "ok", data

    def respond():
        data, status = post(
            f"/api/lifeguards/{state['lg_id']}/respond",
            {"alert_id": "smoke_test_alert", "zone": state["zone_id"]},
        )
        assert status == 200, f"respond status {status}: {data}"
        assert "message" in data, data

    def scoped_zones_api():
        data, status = get("/api/zones", headers={"Authorization": f"Bearer {state['token']}"})
        assert status == 200, f"zones status {status}: {data}"
        items = data.get("items") or []
        for item in items:
            assert int(item.get("id", -1)) == int(state["zone_id"]), item
        print(f"       token-scoped zones={len(items)}")

    def logout():
        data, status = post("/api/lifeguards/logout", headers={"Authorization": f"Bearer {state['token']}"})
        assert status == 200, f"logout status {status}: {data}"

    check("B1 Register lifeguard", register_and_login)
    check("B2 GET /api/lifeguards/me (Bearer)", me)
    check("C1 Assign zone", assign_zone)
    check("D1 Scoped lifeguard alerts", scoped_alerts)
    check("D2 Server-side zone filter on /api/zones", scoped_zones_api)
    check("E1 Heartbeat", heartbeat)
    check("E2 Respond to alert", respond)
    check("B3 Logout", logout)

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)

    print("\nAll Phase B–E API checks passed.")


if __name__ == "__main__":
    main()
