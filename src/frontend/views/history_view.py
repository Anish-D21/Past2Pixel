"""
Restoration History View for Past2Pixel Streamlit Application.
Displays authenticated user's persistent processing sessions with inspection,
lifecycle management, and confirmed deletion controls.
"""
import streamlit as st
from ..api_client import api_client, ApiException
from ..state import get_token, is_authenticated, set_active_session, navigate_to
from ..components.styles import render_status_badge


def render_history_view():
    st.markdown('<h1 class="manuscript-title">Restoration Session History</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="manuscript-subtitle">Review, inspect, and manage your isolated manuscript restoration sessions.</p>',
        unsafe_allow_html=True,
    )

    token = get_token()

    if not is_authenticated() or not token:
        st.warning("🔒 You must be logged in to access and manage your restoration history.")
        col_l, col_s = st.columns(2)
        with col_l:
            if st.button("Sign In to Existing Account", use_container_width=True):
                navigate_to("Login")
                st.rerun()
        with col_s:
            if st.button("Create New Account", use_container_width=True):
                navigate_to("Sign Up")
                st.rerun()
        return

    # -------------------------------------------------------------
    # 1. Fetch User Sessions from Backend
    # -------------------------------------------------------------
    try:
        with st.spinner("Fetching persistent restoration sessions from backend..."):
            sessions = api_client.list_sessions(token)
    except ApiException as e:
        st.error(f"Failed to load session history: {e.detail}")
        return

    if not sessions:
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem 1rem; background: #fafafa; border: 1px dashed #cbd5e1; border-radius: 8px;">
                <h3 style="color: #475569; margin-bottom: 0.5rem;">No Processing Sessions Found</h3>
                <p style="color: #64748b; margin-bottom: 1.5rem;">You haven't uploaded or restored any manuscript folios yet.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("🚀 Restore Your First Manuscript Folio", type="primary", use_container_width=True):
            navigate_to("Restore Manuscript")
            st.rerun()
        return

    st.markdown(f"**Found `{len(sessions)}` authenticated restoration session(s):**")
    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. Render Session Cards
    # -------------------------------------------------------------
    for session in sessions:
        sid = session.get("session_id")
        created = session.get("created_at", "")[:19].replace("T", " ")
        status = session.get("status", "UNKNOWN")
        fname = session.get("filename") or "manuscript_folio.png"
        dmg = session.get("damaged_percentage")
        dmg_str = f"{dmg:.1f}% area" if dmg is not None else "Uncalculated"
        has_ocr = session.get("has_ocr", False)
        has_rest = session.get("has_restoration", False)

        with st.container():
            st.markdown(
                f"""
                <div class="p2p-card" style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div>
                            <span style="font-weight: 700; font-size: 1.05rem;">{fname}</span>
                            <span style="color: #64748b; font-size: 0.85rem; margin-left: 0.5rem;"><code>{sid}</code></span>
                        </div>
                        <div>
                            {render_status_badge(status)}
                        </div>
                    </div>
                    <div style="display: flex; gap: 1.5rem; font-size: 0.85rem; color: #475569;">
                        <div>🕒 <strong>Created:</strong> {created}</div>
                        <div>🔍 <strong>Damage:</strong> {dmg_str}</div>
                        <div>📝 <strong>OCR:</strong> {'Extracted' if has_ocr else 'Not run'}</div>
                        <div>✨ <strong>Restoration:</strong> {'Completed' if has_rest else 'Pending'}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Actions row
            bcol1, bcol2, bcol3 = st.columns([2, 2, 2])
            with bcol1:
                if st.button("📊 Results & Metrics", key=f"hist_res_{sid}", use_container_width=True):
                    set_active_session(sid, redirect_page="Results & Analysis")
                    st.rerun()
            with bcol2:
                if st.button("🔬 Pipeline Stages", key=f"hist_pip_{sid}", use_container_width=True):
                    set_active_session(sid, redirect_page="Pipeline Explanation")
                    st.rerun()
            with bcol3:
                # Deletion with confirmation expander
                with st.expander("🗑️ Delete"):
                    st.caption("This action is permanent and removes all disk artifacts.")
                    if st.button("Confirm Delete", key=f"del_confirm_{sid}", type="primary", use_container_width=True):
                        try:
                            api_client.delete_session(sid, token)
                            st.success(f"Session '{sid}' deleted successfully.")
                            st.rerun()
                        except ApiException as e:
                            st.error(f"Deletion failed: {e.detail}")

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
