"""
Past2Pixel — Ancient Manuscript AI Restoration & Explainable Web Application.
Main Streamlit Application Entrypoint.
"""
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frontend.state import init_session_state
from src.frontend.components.styles import apply_custom_styles
from src.frontend.components.sidebar import render_sidebar
from src.frontend.views.login_view import render_login_view
from src.frontend.views.signup_view import render_signup_view
from src.frontend.views.dashboard_view import render_dashboard_view
from src.frontend.views.restore_view import render_restore_view
from src.frontend.views.ocr_view import render_ocr_view
from src.frontend.views.results_view import render_results_view
from src.frontend.views.history_view import render_history_view
from src.frontend.views.pipeline_view import render_pipeline_view


def main():
    # 1. Page Configuration (Wide academic dashboard layout)
    st.set_page_config(
        page_title="Past2Pixel — Manuscript AI",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Apply Custom Academic Stylesheet
    apply_custom_styles()

    # 3. Initialize Global Session State
    init_session_state()

    # 4. Render Sidebar Navigation & Diagnostic Telemetry
    active_page = render_sidebar()

    # 5. Route to Active Page View
    if active_page == "Login":
        render_login_view()
    elif active_page == "Sign Up":
        render_signup_view()
    elif active_page == "Home Dashboard":
        render_dashboard_view()
    elif active_page == "Restore Manuscript":
        render_restore_view()
    elif active_page == "OCR & Text Extraction":
        render_ocr_view()
    elif active_page == "Results & Analysis":
        render_results_view()
    elif active_page == "Restoration History":
        render_history_view()
    elif active_page == "Pipeline Explanation":
        render_pipeline_view()
    else:
        render_dashboard_view()


if __name__ == "__main__":
    main()
