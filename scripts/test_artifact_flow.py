"""
End-to-End Artifact Generation & HTTP 200 Verification Test.
1. Upload a fresh manuscript folio.
2. Run RestorationEngine via /api/restore.
3. Verify that original.png, restored_unet.png, damage_overlay.png, and damage_mask.png
   exist on disk in the exact session directory.
4. Verify HTTP GET requests for each artifact against both the FastAPI backend
   (http://127.0.0.1:8000) and the Vite frontend dev proxy (http://localhost:5174 or 5173).
5. Verify that a nonexistent/missing artifact URL returns HTTP 404 cleanly.
"""
import urllib.request
import json
import io
from pathlib import Path
from PIL import Image

BASE_BACKEND = 'http://127.0.0.1:8000'
# Detect frontend port (5174 or 5173)
FRONTEND_PORTS = [5174, 5173]

def run_test():
    print("=" * 70)
    print("PAST2PIXEL ARTIFACT RESOLUTION & HTTP 200 VERIFICATION")
    print("=" * 70)

    # 1. Upload fresh image
    img = Image.new('RGB', (256, 256), color=(225, 205, 175))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    boundary = '----ArtifactTestBoundary'
    parts = [
        f'--{boundary}'.encode('ascii'),
        b'Content-Disposition: form-data; name="file"; filename="test_artifacts.png"',
        b'Content-Type: image/png',
        b'',
        buf.getvalue(),
        f'--{boundary}--'.encode('ascii'),
        b''
    ]
    body = b'\r\n'.join(parts)

    up_req = urllib.request.Request(
        f'{BASE_BACKEND}/api/upload',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST'
    )
    with urllib.request.urlopen(up_req) as resp:
        up_data = json.loads(resp.read().decode('utf-8'))
        session_id = up_data['session_id']
        print(f"1. Uploaded new folio -> Session ID: {session_id}")

    # 2. Run Restoration
    res_req = urllib.request.Request(
        f'{BASE_BACKEND}/api/restore',
        data=json.dumps({'session_id': session_id, 'device_preference': 'cpu'}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(res_req) as resp:
        res_data = json.loads(resp.read().decode('utf-8'))
        artifacts = res_data['artifacts']
        print("2. Restoration Pipeline executed successfully!")
        print("   Returned Artifact URLs:")
        for k, v in artifacts.items():
            print(f"     - {k}: {v}")

    # 3. Check Files on Disk
    session_dir = Path("data/sessions") / session_id
    print(f"\n3. Checking files on disk in {session_dir.resolve()}:")
    expected_files = ["original.png", "restored_unet.png", "damage_overlay.png", "damage_mask.png"]
    for fname in expected_files:
        fpath = session_dir / fname
        assert fpath.exists(), f"File {fname} missing from session directory {session_dir}!"
        print(f"   [EXISTS] {fname} ({fpath.stat().st_size} bytes)")

    # 4. HTTP GET Requests via Backend (Port 8000)
    print(f"\n4. Testing HTTP GET via Backend ({BASE_BACKEND}):")
    for name, path in artifacts.items():
        url = f"{BASE_BACKEND}{path}"
        with urllib.request.urlopen(url) as r:
            assert r.status == 200, f"Expected 200 for {url}, got {r.status}"
            content = r.read()
            print(f"   [200 OK] {name} ({url}) -> {len(content)} bytes")

    # 5. HTTP GET Requests via Vite Proxy
    for port in FRONTEND_PORTS:
        host = f"http://localhost:{port}"
        try:
            test_req = urllib.request.Request(f"{host}/")
            with urllib.request.urlopen(test_req):
                frontend_host = host
                break
        except Exception:
            continue
    else:
        frontend_host = None

    if frontend_host:
        print(f"\n5. Testing HTTP GET via Vite Frontend Proxy ({frontend_host}):")
        for name, path in artifacts.items():
            url = f"{frontend_host}{path}"
            with urllib.request.urlopen(url) as r:
                assert r.status == 200, f"Expected 200 for {url}, got {r.status}"
                content = r.read()
                print(f"   [200 OK] {name} ({url}) -> {len(content)} bytes")
    else:
        print("\n5. Vite dev server not detected on tested ports.")

    # 6. Verify Missing/Nonexistent Artifact Handling (Expected 404)
    print("\n6. Testing Nonexistent Artifact Handling:")
    nonexistent_url = f"{BASE_BACKEND}/static/sessions/{session_id}/nonexistent_file.png"
    try:
        urllib.request.urlopen(nonexistent_url)
        assert False, f"Expected 404 for {nonexistent_url}, but request succeeded!"
    except urllib.error.HTTPError as e:
        assert e.code == 404, f"Expected code 404, got {e.code}"
        print(f"   [404 NOT FOUND] {nonexistent_url} correctly returned HTTP 404 as expected.")

    print("\n" + "=" * 70)
    print("ALL ARTIFACT PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_test()
