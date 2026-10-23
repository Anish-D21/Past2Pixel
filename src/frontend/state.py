"""
Session State Management for Past2Pixel Streamlit Application.
Maintains authentication tokens, user profile, and active workflow identifiers.
"""
from typing import Optional, Dict, Any
import streamlit as st


def init_session_state() -> None:
    """Initializes Streamlit session state keys if not already present."""
    if "token" not in st.session_state:
        st.session_state["token"] = None

    if "user" not in st.session_state:
        st.session_state["user"] = None

    if "active_session_id" not in st.session_state:
        st.session_state["active_session_id"] = None

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home Dashboard"

    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None


def is_authenticated() -> bool:
    """Returns True if a valid user token and profile exist in state."""
    init_session_state()
    return bool(st.session_state.get("token") and st.session_state.get("user"))


def set_auth(token: str, user: Dict[str, Any]) -> None:
    """Stores authentication token and user profile in state."""
    st.session_state["token"] = token
    st.session_state["user"] = user


def clear_auth() -> None:
    """Clears user session on logout."""
    st.session_state["token"] = None
    st.session_state["user"] = None
    st.session_state["active_session_id"] = None
    st.session_state["current_page"] = "Login"


def get_token() -> Optional[str]:
    """Retrieves active JWT bearer token."""
    init_session_state()
    return st.session_state.get("token")


def get_user() -> Optional[Dict[str, Any]]:
    """Retrieves authenticated user profile."""
    init_session_state()
    return st.session_state.get("user")


def set_active_session(session_id: str, redirect_page: Optional[str] = None) -> None:
    """Sets the active inspection session and optionally redirects to another page."""
    st.session_state["active_session_id"] = session_id
    if redirect_page:
        st.session_state["current_page"] = redirect_page


def get_active_session_id() -> Optional[str]:
    """Returns currently selected session ID."""
    init_session_state()
    return st.session_state.get("active_session_id")


def navigate_to(page_name: str) -> None:
    """Updates current page view."""
    st.session_state["current_page"] = page_name

