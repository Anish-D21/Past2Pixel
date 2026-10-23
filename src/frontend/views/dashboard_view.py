"""
Home Dashboard View for Past2Pixel Streamlit Application.
Displays authenticated researcher identity, real backend readiness, compute hardware,
session counters, and quick pipeline launchpads.
"""
import streamlit as st
from ..api_client import api_client, ApiException
from ..state import is_authenticated, get_user, get_token, navigate_to, set_active_session
from ..components.styles import render_status_badge


def render_dashboard_view():
    st.markdown('<h1 class="manuscript-title">Research Laboratory Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="manuscript-subtitle">Overview of system readiness, neural restoration engines, persistence telemetry, and user sessions.</p>',
        unsafe_allow_html=True,
    )

    user = get_user()
    token = get_token()

    # -------------------------------------------------------------
    # 1. Fetch Live Diagnostics & User Sessions
    # -------------------------------------------------------------
    readiness = {}
    backend_online = False
    try:
        readiness = api_client.get_readiness()
        backend_online = True
    except ApiException:
        backend_online = False

    user_sessions = []
    if is_authenticated() and token:
        try:
            user_sessions = api_client.list_sessions(token)
        except ApiException:
            user_sessions = []

    # -------------------------------------------------------------
    # 2. Top Metric Cards (Strictly Real Data)
    # -------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="p2p-metric-card">
                <div class="p2p-metric-lbl">Backend Service</div>
                <div class="p2p-metric-val">{render_status_badge('ONLINE' if backend_online else 'OFFLINE')}</div>
                <div style="font-size:0.75rem; color:#6b7280;">FastAPI 1.0.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        db_info = readiness.get("database", {})
        storage_mode = db_info.get("storage_mode", "unknown")
        storage_tag = "ATLAS CLOUD" if storage_mode == "mongodb_atlas" else ("LOCAL JSON" if storage_mode == "local_file_persistence" else "OFFLINE")
        st.markdown(
            f"""
            <div class="p2p-metric-card">
                <div class="p2p-metric-lbl">Persistence Engine</div>
                <div class="p2p-metric-val">{render_status_badge(storage_tag)}</div>
                <div style="font-size:0.75rem; color:#6b7280;">{db_info.get('database_name', 'past2pixel')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        models = readiness.get("models", {})
        unet_stat = models.get("unet_restoration", {}).get("status", "UNTRAINED")
        st.markdown(
            f"""
            <div class="p2p-metric-card">
                <div class="p2p-metric-lbl">Restoration Model</div>
                <div class="p2p-metric-val">{render_status_badge(unet_stat)}</div>
                <div style="font-size:0.75rem; color:#6b7280;">U-Net Restoration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        session_count_str = str(len(user_sessions)) if is_authenticated() else "—"
        sub_lbl = "Owned Sessions" if is_authenticated() else "Sign in to track"
        st.markdown(
            f"""
            <div class="p2p-metric-card">
                <div class="p2p-metric-lbl">My Processing Runs</div>
                <div class="p2p-metric-val">{session_count_str}</div>
                <div style="font-size:0.75rem; color:#6b7280;">{sub_lbl}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. Model Readiness & Compute Disclosures
    # -------------------------------------------------------------
    compute = readiness.get("compute", {})
    dev_name = compute.get("device_name") or f"CPU ({compute.get('cpu_count_logical', 1)} logical cores)"
    dev_type = compute.get("device_type", "cpu").upper()
    ocr = readiness.get("ocr", {})

    st.markdown("### System Telemetry & Hardware Detection")
    st.markdown(
        f"""
        <div class="p2p-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div>
                    <strong>Compute Acceleration:</strong> {dev_type} — {dev_name}
                </div>
                <div>
                    Python {compute.get('python_version', '3.11')} | PyTorch {compute.get('pytorch_version', '2.x')}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.85rem;">
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <strong>Neural Checkpoint Audit:</strong><br/>
                    • U-Net Restoration: {render_status_badge(models.get('unet_restoration', {}).get('status', 'UNTRAINED'))}<br/>
                    • U-Net Damage Detector: {render_status_badge(models.get('unet_damage_detector', {}).get('status', 'UNTRAINED'))}<br/>
                    • Convolutional Autoencoder: {render_status_badge(models.get('autoencoder_baseline', {}).get('status', 'UNTRAINED'))}
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <strong>OCR Linguistic Capabilities:</strong><br/>
                    • Engine Detection: {render_status_badge(ocr.get('status', 'UNAVAILABLE'))}<br/>
                    • Devanagari / Sanskrit Packs: {'Supported' if ocr.get('devanagari_support') else 'Not detected'}<br/>
                    • Available Languages: {', '.join(ocr.get('available_languages', [])) or 'None'}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # 4. Quick Workflow Actions
    # -------------------------------------------------------------
    st.markdown("### Quick Pipeline Launchpad")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)

    with qcol1:
        if st.button("📜 Restore Manuscript", use_container_width=True):
            navigate_to("Restore Manuscript")
            st.rerun()
    with qcol2:
        if st.button("🔍 OCR Text Extraction", use_container_width=True):
            navigate_to("OCR & Text Extraction")
            st.rerun()
    with qcol3:
        if st.button("⏳ View History", use_container_width=True):
            navigate_to("Restoration History")
            st.rerun()
    with qcol4:
        if st.button("🔬 Pipeline Laboratory", use_container_width=True):
            navigate_to("Pipeline Explanation")
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. Recent Processing Sessions (User-isolated)
    # -------------------------------------------------------------
    st.markdown("### Recent Processing Sessions")
    if not is_authenticated():
        st.info("💡 You are currently browsing as a guest. Please sign in to save, inspect, and delete restoration sessions.")
    elif not user_sessions:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem; background: #fafafa; border: 1px dashed #cbd5e1; border-radius: 8px;">
                <p style="color: #64748b; margin-bottom: 1rem;">No manuscript processing sessions found in your account.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Upload First Manuscript Folio", type="primary"):
            navigate_to("Restore Manuscript")
            st.rerun()
    else:
        # Show recent 5 sessions
        recent = user_sessions[:5]
        for s in recent:
            sid = s.get("session_id")
            fname = s.get("filename") or "manuscript_folio.png"
            dmg = s.get("damaged_percentage")
            dmg_str = f"{dmg:.1f}% damaged" if dmg is not None else "Damage unanalyzed"
            s_stat = s.get("status", "UPLOADED")

            sc1, sc2, sc3, sc4 = st.columns([3, 2, 2, 2])
            with sc1:
                st.markdown(f"**`{sid}`** — *{fname}*")
            with sc2:
                st.caption(f"Status: {s_stat} | {dmg_str}")
            with sc3:
                if st.button("🔍 Analyze", key=f"dash_an_{sid}"):
                    set_active_session(sid, redirect_page="Results & Analysis")
                    st.rerun()
            with sc4:
                if st.button("🔬 Pipeline", key=f"dash_pip_{sid}"):
                    set_active_session(sid, redirect_page="Pipeline Explanation")
                    st.rerun()
