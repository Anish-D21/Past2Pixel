"""
Test Suite for User Authentication, Password Security, and Token Authorization.
"""
import pytest
import time
from fastapi.testclient import TestClient

from src.api.app import app
from src.auth.security import hash_password, verify_password, create_access_token, decode_access_token

client = TestClient(app)


def test_password_hashing():
    pwd = "StudentSecret2026!#"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed)
    assert not verify_password("WrongPassword", hashed)


def test_jwt_token_encode_decode():
    payload = {"sub": "usr_999", "email": "test@domain.com"}
    token = create_access_token(payload)
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "usr_999"
    assert decoded["email"] == "test@domain.com"


def test_user_registration_and_login():
    unique_ts = int(time.time() * 1000)
    email = f"auth_test_{unique_ts}@past2pixel.org"
    username = f"user_{unique_ts}"
    password = "StrongPassword123!"

    # 1. Register
    reg_res = client.post("/api/auth/register", json={
        "email": email,
        "username": username,
        "password": password,
    })
    assert reg_res.status_code == 201
    data = reg_res.json()
    assert "access_token" in data
    assert data["user"]["email"] == email

    # 2. Duplicate registration rejected
    dup_res = client.post("/api/auth/register", json={
        "email": email,
        "username": f"other_{username}",
        "password": password,
    })
    assert dup_res.status_code == 400
    assert "already registered" in dup_res.json()["detail"]

    # 3. Login with correct credentials
    login_res = client.post("/api/auth/login", json={
        "username_or_email": email,
        "password": password,
    })
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]

    # 4. Login with bad password rejected
    bad_login = client.post("/api/auth/login", json={
        "username_or_email": email,
        "password": "WrongPassword!",
    })
    assert bad_login.status_code == 401

    # 5. Access /api/auth/me with valid token
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == email

    # 6. Access /api/auth/me without token rejected
    unauth_res = client.get("/api/auth/me")
    assert unauth_res.status_code == 401
