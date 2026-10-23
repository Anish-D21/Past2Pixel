"""
Test Suite for Database Persistence, Session Ownership, and User Isolation.
"""
import io
import time
from PIL import Image
from fastapi.testclient import TestClient

from src.api.app import app
from src.db.mongodb import db_manager

client = TestClient(app)


def test_database_status():
    status = db_manager.get_status()
    assert "status" in status
    assert "storage_mode" in status
    assert status["is_persistent"] is True


def test_session_ownership_and_isolation():
    ts = int(time.time() * 1000)

    # 1. Register User A
    user_a_res = client.post("/api/auth/register", json={
        "email": f"usera_{ts}@domain.com",
        "username": f"user_a_{ts}",
        "password": "Password123!",
    })
    token_a = user_a_res.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # 2. Register User B
    user_b_res = client.post("/api/auth/register", json={
        "email": f"userb_{ts}@domain.com",
        "username": f"user_b_{ts}",
        "password": "Password123!",
    })
    token_b = user_b_res.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 3. User A uploads a manuscript
    img = Image.new("RGB", (256, 256), color=(240, 220, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    upload_res = client.post(
        "/api/upload",
        files={"file": ("usera_folio.png", buf, "image/png")},
        headers=headers_a,
    )
    assert upload_res.status_code == 201
    session_id = upload_res.json()["session_id"]

    # 4. User A can view own session
    view_a = client.get(f"/api/sessions/{session_id}", headers=headers_a)
    assert view_a.status_code == 200
    assert view_a.json()["session_id"] == session_id

    # 5. User B CANNOT view User A's session (Strict Isolation -> 403 Forbidden)
    view_b = client.get(f"/api/sessions/{session_id}", headers=headers_b)
    assert view_b.status_code == 403
    assert "Access denied" in view_b.json()["detail"]

    # 6. User B CANNOT delete User A's session -> 403 Forbidden
    del_b = client.delete(f"/api/sessions/{session_id}", headers=headers_b)
    assert del_b.status_code == 403

    # 7. User A deletes own session -> 200 OK
    del_a = client.delete(f"/api/sessions/{session_id}", headers=headers_a)
    assert del_a.status_code == 200

    # 8. Session no longer exists
    view_again = client.get(f"/api/sessions/{session_id}", headers=headers_a)
    assert view_again.status_code == 404


def test_session_report_downloads_and_isolation():
    ts = int(time.time() * 1000)

    # 1. Register User A and User B
    user_a = client.post("/api/auth/register", json={
        "email": f"rep_a_{ts}@domain.com",
        "username": f"rep_a_{ts}",
        "password": "Password123!",
    }).json()
    token_a = user_a["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    user_b = client.post("/api/auth/register", json={
        "email": f"rep_b_{ts}@domain.com",
        "username": f"rep_b_{ts}",
        "password": "Password123!",
    }).json()
    token_b = user_b["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 2. User A uploads folio
    img = Image.new("RGB", (256, 256), color=(240, 220, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    upload_res = client.post(
        "/api/upload",
        files={"file": ("report_folio.png", buf, "image/png")},
        headers=headers_a,
    )
    assert upload_res.status_code == 201
    sid = upload_res.json()["session_id"]

    # 3. Test report downloads in all 4 formats for User A
    for fmt in ["html", "json", "markdown", "pdf"]:
        rep = client.get(f"/api/sessions/{sid}/report?format={fmt}", headers=headers_a)
        assert rep.status_code == 200
        assert len(rep.content) > 0
        if fmt == "pdf":
            assert rep.content.startswith(b"%PDF")
        elif fmt == "html":
            assert b"<!DOCTYPE html>" in rep.content

    # 4. User B cannot download User A's report (403 Forbidden)
    rep_b = client.get(f"/api/sessions/{sid}/report?format=pdf", headers=headers_b)
    assert rep_b.status_code == 403

    # 5. Nonexistent session report -> 404
    rep_404 = client.get("/api/sessions/nonexistent_sid/report?format=pdf", headers=headers_a)
    assert rep_404.status_code == 404

