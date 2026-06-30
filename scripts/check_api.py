import requests

def check(url):
    try:
        r = requests.get(url, timeout=5)
        print(url, r.status_code)
        try:
            print(r.json())
        except Exception:
            print(r.text[:400])
    except Exception as e:
        print(url, 'ERROR', e)

if __name__ == '__main__':
    for u in ['http://127.0.0.1:8000/api/zones', 'http://127.0.0.1:8000/api/lifeguards']:
        check(u)
