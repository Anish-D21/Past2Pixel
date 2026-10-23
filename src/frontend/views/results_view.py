"""
Results & Scientific Analysis View for Past2Pixel Streamlit Application.
Provides deep inspection of processing artifacts, model readiness disclosures,
reference-dependent metrics (PSNR, SSIM, IoU, Dice), and report downloads.
"""
import streamlit as st
from ..api_client import api_client, ApiException
from ..state import get_token, get_active_session_id, set_active_session, is_authenticated
from ..components.styles import render_status_badge
from ..components.report_builder import generate_session_markdown_report, generate_session_json_export


def render_results_view():
    st.markdown('<h1 class="manuscript-title">Results & Scientific Analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="manuscript-subtitle">Comprehensive analytical inspection of damage localization, neural reconstruction, and quantitative metrics.</p>',
        unsafe_allow_html=True,
    )

    token = get_token()
    active_sid = get_active_session_id()

    # -------------------------------------------------------------
    # 1. Session Selector
    # -------------------------------------------------------------
    user_sessions = []
    if is_authenticated() and token:
        try:
            user_sessions = api_client.list_sessions(token)
        except ApiException:
            user_sessions = []

    session_id_options = [s["session_id"] for s in user_sessions]
    
    # Pre-select active_sid if available
    default_idx = 0
    if active_sid and active_sid in session_id_options:
        default_idx = session_id_options.index(active_sid)

    sel_col1, sel_col2 = st.columns([3, 1])
    with sel_col1:
        if session_id_options:
            selected_sid = st.selectbox(
                "Select Manuscript Session",
                options=session_id_options,
                index=default_idx,
                help="Choose from your authenticated restoration sessions.",
            )
        else:
            selected_sid = st.text_input(
                "Enter Session ID",
                value=active_sid or "",
                placeholder="e.g. p2p_xxxxxxxxxxxx_1792515000",
            )
    with sel_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_btn = st.button("Load Session", use_container_width=True)

    if not selected_sid:
        st.info("Please select or enter a valid session ID to inspect results.")
        return

    # Update active session
    set_active_session(selected_sid)

    # -------------------------------------------------------------
    # 2. Fetch Full Session Document from Backend
    # -------------------------------------------------------------
    try:
        session = api_client.get_session(selected_sid, token=token)
    except ApiException as e:
        if e.status_code == 404:
            st.error(f"Session '{selected_sid}' was not found in the database.")
        elif e.status_code == 403:
            st.error("Access denied: You do not have permission to view this session.")
        else:
            st.error(f"Failed to retrieve session: {e.detail}")
        return

    # Extract session sections
    orig_meta = session.get("original_metadata", {})
    rest_res = session.get("restoration_result") or {}
    ocr_res = session.get("ocr_result") or {}
    predictions = session.get("text_predictions") or []
    pers_info = session.get("persistence_info", {})
    artifacts = session.get("artifacts", {})

    damage = rest_res.get("damage_detection", {})
    restoration = rest_res.get("restoration", {})
    timings = rest_res.get("timings_ms", {})
    readiness = rest_res.get("model_readiness", {})
    dev_info = rest_res.get("device_info", {})

    # -------------------------------------------------------------
    # 3. Session Header & Persistence Summary
    # -------------------------------------------------------------
    st.markdown(
        f"""
        <div class="p2p-card">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem; margin-bottom: 0.75rem;">
                <div>
                    <span style="font-size: 1.1rem; font-weight: 700;">Session: <code>{selected_sid}</code></span>
                    <span style="margin-left: 0.5rem;">{render_status_badge(session.get('status', 'UNKNOWN'))}</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748b;">
                    Created: {session.get('created_at', 'Unknown')[:19]}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; font-size: 0.85rem;">
                <div><strong>Original File:</strong> {orig_meta.get('original_filename', 'folio.png')}</div>
                <div><strong>Dimensions:</strong> {orig_meta.get('width', 'N/A')} × {orig_meta.get('height', 'N/A')} px</div>
                <div><strong>Storage:</strong> {render_status_badge(pers_info.get('storage_mode', 'local_file_persistence'))}</div>
                <div><strong>Device:</strong> {dev_info.get('device_type', 'cpu').upper()}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # 4. Visual Artifacts Inspection Gallery
    # -------------------------------------------------------------
    st.markdown("### 1. Visual Artifact Inspection")
    base_url = api_client.base_url

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("##### Original Folio")
        orig_url = artifacts.get("original")
        if orig_url:
            st.image(f"{base_url}{orig_url}", caption="Raw Input Scan", use_container_width=True)
        else:
            st.caption("Original image not recorded.")

    with c2:
        st.markdown("##### Damage Mask (Binary)")
        mask_url = artifacts.get("damage_mask")
        if mask_url:
            st.image(f"{base_url}{mask_url}", caption="U-Net Binary Mask", use_container_width=True)
        else:
            st.caption("Damage mask unavailable.")

    with c3:
        st.markdown("##### Damage Localization Overlay")
        overlay_url = artifacts.get("damage_overlay")
        if overlay_url:
            st.image(f"{base_url}{overlay_url}", caption=f"Overlay ({damage.get('damaged_pixel_percentage', 0):.1f}% damaged)", use_container_width=True)
        else:
            st.caption("Overlay unavailable.")

    with c4:
        st.markdown("##### Restored Folio")
        rest_url = artifacts.get("restored")
        if rest_url:
            st.image(f"{base_url}{rest_url}", caption="U-Net Restored Output", use_container_width=True)
        else:
            st.caption("Restoration artifact unavailable.")

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. Scientific Metrics Evaluation (Adhering to Academic Integrity)
    # -------------------------------------------------------------
    st.markdown("### 2. Quantitative Evaluation & Metrics")
    st.caption("Scientific restoration and segmentation metrics are evaluated exclusively when authentic ground-truth references exist.")

    sm1, sm2, sm3, sm4 = st.columns(4)

    # PSNR & SSIM metrics
    unet_metrics = restoration.get("unet_metrics", {})
    ref_avail = unet_metrics.get("reference_available", False)

    with sm1:
        if ref_avail:
            st.metric("PSNR (Peak SNR)", f"{unet_metrics.get('psnr_db', 0):.2f} dB")
        else:
            st.markdown(
                f"""
                <div class="p2p-metric-card">
                    <div class="p2p-metric-lbl">PSNR Quality</div>
                    <div style="font-size:0.85rem; font-weight:600; color:#0369a1; margin:0.4rem 0;">{render_status_badge('REFERENCE UNAVAILABLE')}</div>
                    <div style="font-size:0.7rem; color:#6b7280;">Requires clean ground-truth folio</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with sm2:
        if ref_avail:
            st.metric("SSIM (Structural Similarity)", f"{unet_metrics.get('ssim', 0):.4f}")
        else:
            st.markdown(
                f"""
                <div class="p2p-metric-card">
                    <div class="p2p-metric-lbl">SSIM Metric</div>
                    <div style="font-size:0.85rem; font-weight:600; color:#0369a1; margin:0.4rem 0;">{render_status_badge('REFERENCE UNAVAILABLE')}</div>
                    <div style="font-size:0.7rem; color:#6b7280;">Requires clean ground-truth folio</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # IoU & Dice segmentation metrics
    dmg_metrics = damage.get("metrics", {})
    mask_avail = dmg_metrics.get("metrics_available", False)

    with sm3:
        if mask_avail:
            st.metric("Damage IoU (Jaccard)", f"{dmg_metrics.get('iou', 0):.4f}")
        else:
            st.markdown(
                f"""
                <div class="p2p-metric-card">
                    <div class="p2p-metric-lbl">Damage IoU</div>
                    <div style="font-size:0.85rem; font-weight:600; color:#0369a1; margin:0.4rem 0;">{render_status_badge('GROUND TRUTH UNAVAILABLE')}</div>
                    <div style="font-size:0.7rem; color:#6b7280;">Requires annotated damage mask</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with sm4:
        if mask_avail:
            st.metric("Dice Similarity", f"{dmg_metrics.get('dice', 0):.4f}")
        else:
            st.markdown(
                f"""
                <div class="p2p-metric-card">
                    <div class="p2p-metric-lbl">Dice Coefficient</div>
                    <div style="font-size:0.85rem; font-weight:600; color:#0369a1; margin:0.4rem 0;">{render_status_badge('GROUND TRUTH UNAVAILABLE')}</div>
                    <div style="font-size:0.7rem; color:#6b7280;">Requires annotated damage mask</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # 6. Model Readiness & Execution Profile
    # -------------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### 3. Execution Telemetry & Latencies")

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.metric("Total Latency", f"{timings.get('total_pipeline', 0.0):.1f} ms")
    with t2:
        st.metric("Preprocessing", f"{timings.get('preprocessing', 0.0):.1f} ms")
    with t3:
        st.metric("Damage Localization", f"{timings.get('damage_detection', 0.0):.1f} ms")
    with t4:
        st.metric("U-Net Restoration", f"{timings.get('unet_restoration', 0.0):.1f} ms")

    # -------------------------------------------------------------
    # 7. OCR & Text Predictions
    # -------------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### 4. Text Preservation & Linguistic Reconstructions")

    ocr_col, pred_col = st.columns(2)
    with ocr_col:
        st.markdown("#### OCR Transcription")
        if ocr_res and ocr_res.get("extracted_text"):
            st.markdown(f'<div class="devanagari-text">{ocr_res.get("extracted_text")}</div>', unsafe_allow_html=True)
            st.caption(f"Status: {ocr_res.get('status')} | {ocr_res.get('character_count', 0)} characters | {ocr_res.get('processing_time_ms', 0):.1f} ms")
        else:
            st.info("OCR has not been executed on this session yet.")
            if st.button("Run OCR on this Folio", key="btn_run_ocr_from_results"):
                set_active_session(selected_sid, redirect_page="OCR & Text Extraction")
                st.rerun()

    with pred_col:
        st.markdown("#### Speculative Linguistic Predictions")
        if predictions:
            for p in predictions:
                st.markdown(
                    f"""
                    <div style="background:#faf5ff; border:1px solid #e9d5ff; border-radius:6px; padding:0.75rem; margin-bottom:0.5rem;">
                        <strong>Prediction:</strong> <span style="font-size:1.1rem; color:#6b21a8;">{p.get('prediction', 'N/A')}</span><br/>
                        <span style="font-size:0.75rem; color:#9333ea;">Uncertainty: {p.get('uncertainty_label', 'SPECULATIVE')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No contextual linguistic predictions recorded.")

    # -------------------------------------------------------------
    # 8. Report Generation & Download Options
    # -------------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### 5. Laboratory Technical Report Generation")

    md_report = generate_session_markdown_report(session)
    json_export = generate_session_json_export(session)

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.download_button(
            "📥 Download Academic Markdown Report (.md)",
            data=md_report,
            file_name=f"past2pixel_report_{selected_sid}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with r_col2:
        st.download_button(
            "📥 Download Raw Session Telemetry (.json)",
            data=json_export,
            file_name=f"past2pixel_telemetry_{selected_sid}.json",
            mime="application/json",
            use_container_width=True,
        )
