"""
Sidebar Navigation Component for Past2Pixel Streamlit Application.
Displays system health indicators, navigation buttons, and authentication controls.
"""
import streamlit as st
from ..api_client import api_client, ApiException
from ..state import is_authenticated, get_user, clear_auth, navigate_to
from .styles import render_status_badge


def render_sidebar() -> str:
    """
    Renders academic navigation sidebar and system diagnostic indicators.
    Returns the currently active page name.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <h2 style="font-family: 'Cinzel', serif; margin-bottom: 0px;">PAST2PIXEL</h2>
                <p style="font-size: 0.8rem; color: #6b7280; margin-top: 0px;">Ancient Manuscript AI Restoration</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------
        # User Authentication Status
        # ---------------------------------------------------------
        user = get_user()
        if is_authenticated() and user:
            st.markdown(
                f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Authenticated User</div>
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">{user.get('username', 'User')}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">{user.get('email', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 6px; padding: 0.6rem; margin-bottom: 1rem; font-size: 0.85rem; color: #92400e;">
                    🔒 <strong>Guest Mode</strong>: Please log in or register to process and persist restoration sessions.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------
        # Navigation Options
        # ---------------------------------------------------------
        st.markdown("<p style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 0.5rem;'>Navigation</p>", unsafe_allow_html=True)
        
        pages = [
            "Home Dashboard",
            "Restore Manuscript",
            "OCR & Text Extraction",
            "Results & Analysis",
            "Restoration History",
            "Pipeline Explanation",
        ]

        if not is_authenticated():
            auth_pages = ["Login", "Sign Up"]
            all_nav = auth_pages + pages
        else:
            all_nav = pages

        current_page = st.session_state.get("current_page", "Home Dashboard")
        if current_page not in all_nav:
            current_page = all_nav[0]

        selected_page = st.radio(
            "Go to",
            all_nav,
            index=all_nav.index(current_page),
            label_visibility="collapsed",
            key="sidebar_nav_radio",
        )

        if selected_page != current_page:
            st.session_state.current_page = selected_page
            st.rerun()

        # Logout button
        if is_authenticated():
            st.markdown("<hr style='margin: 1.5rem 0 1rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
            if st.button("🚪 Log Out", use_container_width=True):
                clear_auth()
                st.success("Successfully logged out.")
                st.rerun()

        # ---------------------------------------------------------
        # Live System Health Indicators
        # ---------------------------------------------------------
        st.markdown("<hr style='margin: 1.5rem 0 1rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 0.5rem;'>System Diagnostics</p>", unsafe_allow_html=True)

        try:
            readiness = api_client.get_readiness()
            db_info = readiness.get("database", {})
            compute = readiness.get("compute", {})
            ocr = readiness.get("ocr", {})
            models = readiness.get("models", {})

            # 1. API Status
            st.markdown(f"<div style='font-size:0.8rem; margin-bottom:4px;'>Backend: {render_status_badge('ONLINE')}</div>", unsafe_allow_html=True)

            # 2. Storage Mode
            storage_mode = db_info.get("storage_mode", "unknown")
            storage_label = "ATLAS" if storage_mode == "mongodb_atlas" else "LOCAL JSON"
            st.markdown(f"<div style='font-size:0.8rem; margin-bottom:4px;'>Persistence: {render_status_badge(storage_label)}</div>", unsafe_allow_html=True)

            # 3. Compute
            dev_type = compute.get("device_type", "cpu").upper()
            dev_desc = compute.get("device_name") or f"CPU ({compute.get('cpu_count_logical', 1)} cores)"
            st.markdown(f"<div style='font-size:0.8rem; margin-bottom:4px;'>Compute: <span style='font-weight:600;'>{dev_type}</span> <span style='color:#64748b; font-size:0.75rem;'>({dev_desc[:18]})</span></div>", unsafe_allow_html=True)

            # 4. Checkpoint Status
            unet_stat = models.get("unet_restoration", {}).get("status", "UNTRAINED")
            st.markdown(f"<div style='font-size:0.8rem; margin-bottom:4px;'>Models: {render_status_badge(unet_stat)}</div>", unsafe_allow_html=True)

            # 5. OCR Status
            ocr_stat = ocr.get("status", "UNAVAILABLE")
            st.markdown(f"<div style='font-size:0.8rem;'>OCR Engine: {render_status_badge(ocr_stat)}</div>", unsafe_allow_html=True)

        except ApiException:
            st.markdown(f"<div style='font-size:0.8rem; color:#dc2626;'>Backend: {render_status_badge('OFFLINE')}</div>", unsafe_allow_html=True)
            st.caption("FastAPI server is not responding at " + api_client.base_url)

    return st.session_state.current_page
