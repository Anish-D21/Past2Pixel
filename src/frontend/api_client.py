"""
API Client Layer for Past2Pixel Streamlit Frontend.
Provides strongly-typed, exception-safe HTTP communications with the FastAPI backend.
"""
import os
from typing import Dict, Any, Optional, List, Union
import requests


class ApiException(Exception):
    """Encapsulates backend HTTP error responses with status code and detail message."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class ApiClient:
    """HTTP Client communicating with the Past2Pixel FastAPI backend service."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 45):
        # Configure backend base URL with priority: parameter > env var > default
        self.base_url = (base_url or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout

    def _headers(self, token: Optional[str] = None) -> Dict[str, str]:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token.strip()}"
        return headers

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Parses JSON response or raises ApiException with sanitized server detail."""
        try:
            payload = response.json()
        except Exception:
            payload = {"detail": response.text or "Non-JSON response returned by backend server."}

        if not response.ok:
            detail = payload.get("detail", f"Backend error occurred (HTTP {response.status_code}).")
            if isinstance(detail, list):
                # Pydantic validation error list
                err_messages = [f"{err.get('loc', ['field'])[-1]}: {err.get('msg', 'invalid')}" for err in detail]
                detail = "; ".join(err_messages)
            raise ApiException(status_code=response.status_code, detail=str(detail))

        return payload

    # -------------------------------------------------------------
    # Health & System Diagnostics
    # -------------------------------------------------------------
    def get_health(self) -> Dict[str, Any]:
        """Checks backend availability via /health."""
        try:
            url = f"{self.base_url}/health"
            res = requests.get(url, timeout=5)
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, f"Backend service is unreachable at {self.base_url}. Please start FastAPI.")
        except requests.exceptions.Timeout:
            raise ApiException(504, "Backend health check timed out.")
        except ApiException:
            raise
        except Exception as e:
            raise ApiException(500, f"Health check failed: {e}")

    def get_readiness(self) -> Dict[str, Any]:
        """Fetches complete hardware, model, OCR, and DB readiness statuses."""
        try:
            url = f"{self.base_url}/readiness"
            res = requests.get(url, timeout=10)
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, f"Backend service is unreachable at {self.base_url}.")
        except Exception as e:
            if isinstance(e, ApiException):
                raise
            raise ApiException(500, f"Readiness diagnostic failed: {e}")

    # -------------------------------------------------------------
    # Authentication & Profile
    # -------------------------------------------------------------
    def register(self, email: str, username: str, password: str) -> Dict[str, Any]:
        """Registers a new user account with hashed credentials."""
        url = f"{self.base_url}/api/auth/register"
        data = {
            "email": email.strip(),
            "username": username.strip(),
            "password": password,
        }
        try:
            res = requests.post(url, json=data, timeout=self.timeout)
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, "Cannot reach backend server to complete registration.")

    def login(self, username_or_email: str, password: str) -> Dict[str, Any]:
        """Authenticates user and returns JWT bearer token and user profile."""
        url = f"{self.base_url}/api/auth/login"
        data = {
            "username_or_email": username_or_email.strip(),
            "password": password,
        }
        try:
            res = requests.post(url, json=data, timeout=self.timeout)
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, "Cannot reach backend server to authenticate.")

    def get_current_user(self, token: str) -> Dict[str, Any]:
        """Retrieves profile of the currently authenticated token owner."""
        url = f"{self.base_url}/api/auth/me"
        try:
            res = requests.get(url, headers=self._headers(token), timeout=self.timeout)
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, "Backend connection failed while verifying current user.")

    # -------------------------------------------------------------
    # Manuscript Image Upload
    # -------------------------------------------------------------
    def upload_manuscript(
        self,
        file_bytes: bytes,
        filename: str,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Uploads manuscript folio image to backend to establish a processing session."""
        url = f"{self.base_url}/api/upload"
        # Determine MIME type
        ext = filename.lower().split(".")[-1] if "." in filename else "png"
        mime = f"image/{ext}" if ext in ["png", "jpeg", "jpg", "bmp", "tiff"] else "application/octet-stream"
        
        files = {"file": (filename, file_bytes, mime)}
        try:
            res = requests.post(
                url,
                files=files,
                headers=self._headers(token),
                timeout=self.timeout,
            )
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, "Cannot reach backend server to upload manuscript.")

    # -------------------------------------------------------------
    # Pipeline Execution: Restoration & Damage Localization
    # -------------------------------------------------------------
    def run_restoration(
        self,
        session_id: str,
        device_preference: str = "auto",
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes damage detection and U-Net restoration pipeline on the session image."""
        url = f"{self.base_url}/api/restore"
        payload = {
            "session_id": session_id,
            "device_preference": device_preference,
        }
        try:
            res = requests.post(
                url,
                json=payload,
                headers=self._headers(token),
                timeout=self.timeout * 2,  # allow compute headroom
            )
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, "Cannot reach backend server to execute restoration.")

    # -------------------------------------------------------------
    # OCR & Text Extraction
    # -------------------------------------------------------------
    def run_ocr(
        self,
        session_id: str,
        language: str = "san+hin+eng",
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs OCR extraction on an existing session manuscript."""
        url = f"{self.base_url}/api/ocr"
        payload = {
            "session_id": session_id,
            "language": language,
        }
        try:
            res = requests.post(
                url,
                json=payload,
                headers=self._headers(token),
                timeout=self.timeout,
            )
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, "Cannot reach backend server to execute OCR.")

    def run_direct_ocr(
        self,
        file_bytes: bytes,
        filename: str,
        language: str = "san+hin+eng",
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes direct OCR extraction without requiring an existing session."""
        url = f"{self.base_url}/api/ocr/direct"
        ext = filename.lower().split(".")[-1] if "." in filename else "png"
        mime = f"image/{ext}" if ext in ["png", "jpeg", "jpg", "bmp", "tiff"] else "application/octet-stream"

        files = {"file": (filename, file_bytes, mime)}
        data = {"language": language}
        try:
            res = requests.post(
                url,
                files=files,
                data=data,
                headers=self._headers(token),
                timeout=self.timeout,
            )
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, "Cannot reach backend server to execute direct OCR.")

    # -------------------------------------------------------------
    # Linguistic Context Prediction
    # -------------------------------------------------------------
    def predict_text(
        self,
        left_context: str,
        right_context: str,
        max_candidates: int = 3,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Estimates missing or damaged word spans statistically from surrounding Vedic context."""
        url = f"{self.base_url}/api/predict-text"
        payload = {
            "left_context": left_context,
            "right_context": right_context,
            "max_candidates": max_candidates,
        }
        try:
            res = requests.post(
                url,
                json=payload,
                headers=self._headers(token),
                timeout=self.timeout,
            )
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, "Cannot reach backend server to execute text prediction.")

    # -------------------------------------------------------------
    # Session History & Inspection
    # -------------------------------------------------------------
    def list_sessions(self, token: str) -> List[Dict[str, Any]]:
        """Lists restoration sessions belonging to the authenticated user."""
        url = f"{self.base_url}/api/sessions"
        try:
            res = requests.get(url, headers=self._headers(token), timeout=self.timeout)
            data = self._handle_response(res)
            return data if isinstance(data, list) else []
        except requests.exceptions.ConnectionError:
            raise ApiException(503, "Cannot reach backend server to retrieve sessions.")

    def get_session(self, session_id: str, token: Optional[str] = None) -> Dict[str, Any]:
        """Fetches full image-specific processing session report."""
        url = f"{self.base_url}/api/sessions/{session_id}"
        try:
            res = requests.get(url, headers=self._headers(token), timeout=self.timeout)
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, f"Cannot reach backend server to fetch session '{session_id}'.")

    def delete_session(self, session_id: str, token: str) -> Dict[str, Any]:
        """Permanently deletes a session and its disk artifacts (owner-only)."""
        url = f"{self.base_url}/api/sessions/{session_id}"
        try:
            res = requests.delete(url, headers=self._headers(token), timeout=self.timeout)
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            raise ApiException(503, f"Cannot reach backend server to delete session '{session_id}'.")


# Global default client instance
api_client = ApiClient()
