import sys
import traceback

sys.path.insert(0, 'backend')

try:
    from backend import server
    with server.app.test_client() as client:
        resp = client.get('/api/analytics/crowd-alerts?limit=5')
        print('status_code=', resp.status_code)
        print(resp.data.decode('utf-8'))
except Exception:
    traceback.print_exc()
