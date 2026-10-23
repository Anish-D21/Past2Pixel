"""
Manuscript Restoration & Damage Localization View for Past2Pixel.
Handles folio image upload, model readiness auditing, diagnostic inference execution,
and side-by-side visual artifact presentation.
"""
from PIL import Image
import io
import streamlit as st
from ..api_client import api_client, ApiException
from ..state import get_token, set_active_session, navigate_to
from ..components.styles import render_status_badge


def render_restore_view():
    st.markdown('<h1 class="manuscript-title">Manuscript Restoration Engine</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="manuscript-subtitle">Execute deep learning damage localization (U-Net) and image reconstruction on ancient folios.</p>',
        unsafe_allow_html=True,
    )

    token = get_token()

    # 1. Inspect Model Readiness from Backend
    models_ready = False
    try:
        readiness = api_client.get_readiness()
        models = readiness.get("models", {})
        unet_stat = models.get("unet_restoration", {}).get("status", "UNTRAINED")
        models_ready = unet_stat == "READY_FOR_INFERENCE"
    except Exception:
        unet_stat = "UNTRAINED"

    # 2. Critical Academic Disclosure Banner
    if not models_ready:
        st.markdown(
            f"""
            <div class="p2p-notice-warning">
                <strong>⚠️ Academic Evaluation Disclosure — Untrained Model Checkpoint:</strong><br/>
                The deep learning restoration and damage detection models are currently running with initial random weights (<code>{unet_stat}</code>).
                Executing the pipeline below will run the <strong>authentic forward tensor pass</strong>, produce intermediate feature maps, and compute execution latencies,
                but the visual outputs reflect random initialization and <strong>must not be presented as scientifically verified restorations</strong>.
                Training checkpoints on the GPU cluster will be performed in subsequent phases.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. File Upload & Specifications
    st.markdown("### 1. Upload Manuscript Folio")
    st.caption("Supported formats: PNG, JPG, JPEG, TIFF, BMP | Maximum payload size: 15 MB")

    uploaded_file = st.file_uploader(
        "Choose a manuscript image",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        help="Upload an ancient manuscript scan (Vedic Sanskrit, Devanagari, or historical palm-leaf folio).",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            w, h = pil_img.size
            mode = pil_img.mode
            fmt = pil_img.format or "PNG"
            file_kb = len(file_bytes) / 1024
        except Exception as e:
            st.error(f"Cannot decode image file: {e}")
            return

        # Display upload details
        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; margin-bottom: 1rem; font-size: 0.85rem;">
                <strong>Folio Specifications:</strong> {file_name} | {w} × {h} px | {mode} color mode | {fmt} | {file_kb:.1f} KB
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_prev, col_exec = st.columns([1, 1])
        with col_prev:
            st.image(pil_img, caption="Original Input Manuscript Folio", use_container_width=True)

        with col_exec:
            st.markdown("### 2. Execution Parameters")
            device_choice = st.selectbox(
                "Hardware Acceleration Device",
                options=["auto", "cpu", "cuda"],
                index=0,
                help="Select compute device. 'auto' selects CUDA if available, falling back cleanly to CPU.",
            )

            action_label = "Run Diagnostic Pipeline (Untrained Checkpoint)" if not models_ready else "Execute Neural Restoration Pipeline"
            run_btn = st.button(action_label, type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("1/2 Uploading and establishing manuscript session..."):
                    try:
                        upload_res = api_client.upload_manuscript(
                            file_bytes=file_bytes,
                            filename=file_name,
                            token=token,
                        )
                        session_id = upload_res["session_id"]
                    except ApiException as e:
                        st.error(f"Upload failed: {e.detail}")
                        return

                with st.spinner("2/2 Executing U-Net Damage Localization & Restoration Engine..."):
                    try:
                        rest_res = api_client.run_restoration(
                            session_id=session_id,
                            device_preference=device_choice,
                            token=token,
                        )
                        set_active_session(session_id)
                        st.session_state["last_restoration_result"] = rest_res
                        st.success(f"Processing session **`{session_id}`** executed successfully!")
                    except ApiException as e:
                        st.error(f"Restoration pipeline execution failed: {e.detail}")
                        return

    # -------------------------------------------------------------
    # 4. Display Processing Results if Available
    # -------------------------------------------------------------
    res_data = st.session_state.get("last_restoration_result")
    if res_data and uploaded_file is not None:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### 3. Pipeline Output & Intermediate Artifacts")

        sid = res_data.get("session_id")
        damage = res_data.get("damage_detection", {})
        restoration = res_data.get("restoration", {})
        timings = res_data.get("timings_ms", {})
        dev_info = res_data.get("device_info", {})

        # Latency metric pills
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            dmg_pct = damage.get("damaged_pixel_percentage", 0.0)
            st.metric("Detected Damage", f"{dmg_pct:.1f}%")
        with m2:
            st.metric("Total Latency", f"{timings.get('total_pipeline', 0.0):.1f} ms")
        with m3:
            st.metric("U-Net Latency", f"{timings.get('unet_restoration', 0.0):.1f} ms")
        with m4:
            st.metric("Execution Device", dev_info.get("device_type", "cpu").upper())

        # Triple visual comparison
        c1, c2, c3 = st.columns(3)
        base_url = api_client.base_url

        with c1:
            st.markdown("#### Original Folio")
            orig_url = f"{base_url}/static/sessions/{sid}/original.png"
            st.image(orig_url, caption="Input Folio Scan", use_container_width=True)

        with c2:
            st.markdown("#### Damage Localization")
            overlay_url = f"{base_url}/static/sessions/{sid}/damage_overlay.png"
            st.image(overlay_url, caption=f"U-Net Damage Overlay ({dmg_pct:.1f}% area)", use_container_width=True)

        with c3:
            st.markdown("#### U-Net Restored Folio")
            rest_url = f"{base_url}/static/sessions/{sid}/restored_unet.png"
            st.image(
                rest_url,
                caption="Restored Output (Untrained Forward Pass)" if not models_ready else "Trained U-Net Restoration",
                use_container_width=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        ncol1, ncol2 = st.columns(2)
        with ncol1:
            if st.button("📊 View Full Metrics & Scientific Analysis", use_container_width=True):
                set_active_session(sid, redirect_page="Results & Analysis")
                st.rerun()
        with ncol2:
            if st.button("🔬 Inspect Laboratory Pipeline Stages", use_container_width=True):
                set_active_session(sid, redirect_page="Pipeline Explanation")
                st.rerun()
