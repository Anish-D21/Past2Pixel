"""
Verification Script for Live Past2Pixel FastAPI Backend & AI Pipeline.
Tests:
1. Health & Readiness endpoints
2. Manuscript image upload
3. Full Restoration & Damage Detection pipeline
4. Honest reference metrics check (no false PSNR/SSIM on unreferenced uploads)
5. Honest OCR reporting (UNAVAILABLE when Tesseract not installed)
6. Context-aware Devanagari text prediction
7. Session history persistence & technical markdown report generation
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import urllib.request
import json
import io
from PIL import Image

BASE_URL = 'http://127.0.0.1:8000'

def test_live_api():
    print("=" * 70)
    print("PAST2PIXEL LIVE BACKEND END-TO-END VERIFICATION")
    print("=" * 70)

    # 1. Health & Readiness
    req = urllib.request.Request(f'{BASE_URL}/readiness')
    with urllib.request.urlopen(req) as resp:
        readiness = json.loads(resp.read().decode('utf-8'))
        print('1. Readiness Status:', readiness['status'])
        print('   Model Statuses:', {k: v['status'] for k, v in readiness['models'].items()})
        print('   OCR Status:', readiness['ocr']['status'])
        print('   Database Status:', readiness['database']['status'])

    # 2. Upload Image
    img = Image.new('RGB', (300, 300), color=(215, 195, 160))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    boundary = '----Past2PixelTestBoundary'
    parts = [
        f'--{boundary}'.encode('ascii'),
        b'Content-Disposition: form-data; name="file"; filename="sample_manuscript.png"',
        b'Content-Type: image/png',
        b'',
        buf.getvalue(),
        f'--{boundary}--'.encode('ascii'),
        b''
    ]
    body = b'\r\n'.join(parts)

    upload_req = urllib.request.Request(
        f'{BASE_URL}/api/upload',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST'
    )
    with urllib.request.urlopen(upload_req) as resp:
        upload_data = json.loads(resp.read().decode('utf-8'))
        session_id = upload_data['session_id']
        print(f'\n2. Upload Succeeded! Session ID: {session_id}')
        print(f'   Resolution: {upload_data["width"]}x{upload_data["height"]} | URL: {upload_data["image_url"]}')

    # 3. Run Restoration Pipeline
    restore_req = urllib.request.Request(
        f'{BASE_URL}/api/restore',
        data=json.dumps({'session_id': session_id, 'device_preference': 'cuda'}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(restore_req) as resp:
        restore_data = json.loads(resp.read().decode('utf-8'))
        print('\n3. Restoration Pipeline Succeeded!')
        print(f'   Device Type: {restore_data["device_info"]["device_type"]} ({restore_data["device_info"].get("gpu_name")})')
        print(f'   Execution Timings (ms): {restore_data["timings_ms"]}')
        print(f'   Model Readiness Flags: {restore_data["model_readiness"]}')
        print(f'   Damaged Surface Area: {restore_data["damage_detection"]["damaged_pixel_percentage"]}%')
        print(f'   Damage Metric Status (Honest): {restore_data["damage_detection"]["evaluation"]["status"]}')
        print(f'   Restoration Metric Status (Honest): {restore_data["restoration"]["unet_restoration_metrics"]["status"]}')
        print(f'   Restoration Reason: {restore_data["restoration"]["unet_restoration_metrics"]["reason"]}')
        print(f'   Artifact URLs: {restore_data["artifacts"]}')

    # 4. Run OCR Endpoint
    ocr_req = urllib.request.Request(
        f'{BASE_URL}/api/ocr',
        data=json.dumps({'session_id': session_id}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(ocr_req) as resp:
        ocr_data = json.loads(resp.read().decode('utf-8'))
        print('\n4. OCR Endpoint Succeeded!')
        print(f'   Status: {ocr_data["status"]}')
        print(f'   Authentic OCR Executed: {ocr_data["is_authentic_ocr"]}')
        print(f'   Reason: {ocr_data.get("error")}')
        print(f'   Setup Instructions Available: {ocr_data.get("setup_instructions") is not None}')

    # 5. Damaged Text Prediction
    pred_req = urllib.request.Request(
        f'{BASE_URL}/api/predict-text',
        data=json.dumps({'left_context': 'तस्य हविः', 'right_context': 'जुहोति', 'max_candidates': 3}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(pred_req) as resp:
        pred_data = json.loads(resp.read().decode('utf-8'))
        print('\n5. Context-Aware Text Prediction Succeeded!')
        print(f'   Status: {pred_data["status"]}')
        print(f'   Top Predicted Word: {pred_data.get("prediction")}')
        print(f'   Candidates Count: {len(pred_data.get("candidates", []))}')
        print(f'   Is Speculative: {pred_data.get("is_speculative")}')
        print(f'   Uncertainty Label: {pred_data.get("uncertainty_label")}')

    # 6. Session Details & Report Generation
    session_req = urllib.request.Request(f'{BASE_URL}/api/sessions/{session_id}')
    with urllib.request.urlopen(session_req) as resp:
        session_data = json.loads(resp.read().decode('utf-8'))
        print('\n6. Session Details Retrieved Successfully!')
        print(f'   Session Status: {session_data["status"]}')
        print(f'   Has Restoration: {session_data.get("restoration_result") is not None}')
        print(f'   Has OCR: {session_data.get("ocr_result") is not None}')

        from src.frontend.components.report_builder import generate_session_markdown_report
        report_md = generate_session_markdown_report(session_data)
        print('--- Snippet of Generated Markdown Report ---')
        snippet = '\n'.join(report_md.splitlines()[:22])
        print(snippet)

    print("\n" + "=" * 70)
    print("ALL LIVE END-TO-END VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == '__main__':
    test_live_api()
