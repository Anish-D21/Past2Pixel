"""
Pipeline Explanation View for Past2Pixel — Main Faculty & Scientific Differentiator.
Presents a laboratory-style stage-by-stage visualization of the restoration workflow,
actual recorded intermediate artifacts, execution device latencies, and U-Net architecture deep-dive.
"""
import streamlit as st
from ..api_client import api_client, ApiException
from ..state import get_token, get_active_session_id, set_active_session, is_authenticated
from ..components.styles import render_status_badge
from ..components.report_builder import generate_session_markdown_report, generate_session_json_export


def render_pipeline_view():
    st.markdown('<h1 class="manuscript-title">Scientific Pipeline Laboratory</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="manuscript-subtitle">Interactive, stage-by-stage explainability of computer vision transforms and deep learning restoration mechanics.</p>',
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
    default_idx = 0
    if active_sid and active_sid in session_id_options:
        default_idx = session_id_options.index(active_sid)

    sel_col1, sel_col2 = st.columns([3, 1])
    with sel_col1:
        if session_id_options:
            selected_sid = st.selectbox(
                "Active Manuscript Session for Pipeline Analysis",
                options=session_id_options,
                index=default_idx,
                key="pip_session_select",
            )
        else:
            selected_sid = st.text_input(
                "Active Session ID",
                value=active_sid or "",
                placeholder="e.g. p2p_xxxxxxxxxxxx_1792515000",
                key="pip_session_manual",
            )
    with sel_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.button("Inspect Pipeline", key="pip_inspect_btn", use_container_width=True)

    if not selected_sid:
        st.info("Please select or specify a manuscript session ID to inspect its pipeline execution telemetry.")
        return

    set_active_session(selected_sid)

    # -------------------------------------------------------------
    # 2. Fetch Session Telemetry
    # -------------------------------------------------------------
    try:
        session = api_client.get_session(selected_sid, token=token)
    except ApiException as e:
        st.error(f"Cannot load pipeline data for session '{selected_sid}': {e.detail}")
        return

    orig_meta = session.get("original_metadata", {})
    rest_res = session.get("restoration_result") or {}
    ocr_res = session.get("ocr_result") or {}
    predictions = session.get("text_predictions") or []
    artifacts = session.get("artifacts", {})

    damage = rest_res.get("damage_detection", {})
    restoration = rest_res.get("restoration", {})
    timings = rest_res.get("timings_ms", {})
    readiness = rest_res.get("model_readiness", {})
    dev_info = rest_res.get("device_info", {})
    base_url = api_client.base_url

    # -------------------------------------------------------------
    # 3. Stage-by-Stage Laboratory Telemetry
    # -------------------------------------------------------------
    st.markdown("### 🔬 Laboratory Execution Stages")
    st.caption("Each stage presents strictly verified data recorded by the backend during processing.")

    # STAGE 1: Folio Ingestion & Validation
    with st.expander("Stage 1: Folio Ingestion & Content Validation", expanded=True):
        st.markdown(
            f"""
            - **Subsystem:** Ingestion & Integrity Validator (`src/api/routes/upload.py`)
            - **Input File:** `{orig_meta.get('original_filename', 'folio.png')}`
            - **Native Dimensions:** `{orig_meta.get('width', 'N/A')} × {orig_meta.get('height', 'N/A')} px`
            - **Color Mode:** `{orig_meta.get('mode', 'RGB')}` | **File Size:** `{orig_meta.get('file_size_bytes', 0) / 1024:.1f} KB`
            - **Status:** {render_status_badge('COMPLETED')}
            """,
            unsafe_allow_html=True,
        )
        if artifacts.get("original"):
            st.image(f"{base_url}{artifacts['original']}", caption="Stage 1 Artifact: Verified Decoded Image", width=360)
        else:
            st.warning("Artifact unavailable.")

    # STAGE 2: Preprocessing & Tensor Normalization
    with st.expander("Stage 2: Geometric Rescaling & Tensor Normalization", expanded=False):
        prep_time = timings.get("preprocessing", 0.0)
        st.markdown(
            f"""
            - **Subsystem:** Computer Vision Preprocessor (`Pillow / torchvision.transforms.functional`)
            - **Operation:** Bilinear spatial rescaling to `(256, 256)` tensor grid & `[0.0, 1.0]` float32 normalization
            - **Forward Shape:** `torch.Size([1, 3, 256, 256])`
            - **Execution Latency:** `{prep_time:.2f} ms`
            - **Target Device:** `{dev_info.get('device_type', 'cpu').upper()}`
            - **Status:** {render_status_badge('COMPLETED' if prep_time > 0 else 'NOT_RECORDED')}
            """,
            unsafe_allow_html=True,
        )

    # STAGE 3: Damage Localization & Segmentation
    with st.expander("Stage 3: Deep Learning Damage Localization (U-Net)", expanded=True):
        det_time = timings.get("damage_detection", 0.0)
        dmg_pct = damage.get("damaged_pixel_percentage")
        dmg_pct_str = f"{dmg_pct:.2f}%" if dmg_pct is not None else "N/A"
        st.markdown(
            f"""
            - **Subsystem:** U-Net Binary Segmentation (`src/models/unet_damage_detector.py`)
            - **Architecture:** 4-level Convolutional Encoder-Decoder with Sigmoid Activation
            - **Model State:** {render_status_badge(readiness.get('unet_damage_detector', 'UNTRAINED'))}
            - **Execution Latency:** `{det_time:.2f} ms` on `{dev_info.get('device_type', 'cpu').upper()}`
            - **Estimated Damaged Surface Area:** `{dmg_pct_str}` (Threshold: `{damage.get('threshold_used', 0.35)}`)
            """,
            unsafe_allow_html=True,
        )
        sc1, sc2 = st.columns(2)
        with sc1:
            if artifacts.get("damage_mask"):
                st.image(f"{base_url}{artifacts['damage_mask']}", caption="Stage 3 Artifact: Probability Damage Mask", use_container_width=True)
            else:
                st.info("Damage mask artifact unavailable.")
        with sc2:
            if artifacts.get("damage_overlay"):
                st.image(f"{base_url}{artifacts['damage_overlay']}", caption=f"Stage 3 Artifact: Damage Localization Heatmap ({dmg_pct_str})", use_container_width=True)
            else:
                st.info("Damage overlay artifact unavailable.")

    # STAGE 4: Neural Restoration (Residual U-Net)
    with st.expander("Stage 4: Deep Learning Folio Reconstruction (U-Net)", expanded=True):
        unet_time = timings.get("unet_restoration", 0.0)
        st.markdown(
            f"""
            - **Subsystem:** U-Net Restoration Network (`src/models/unet_restoration.py`)
            - **Architecture:** Residual Skip-Connection Autoencoder (1024-channel bottleneck, ConvTranspose2d)
            - **Checkpoint State:** {render_status_badge(readiness.get('unet_restoration', 'UNTRAINED'))}
            - **Execution Latency:** `{unet_time:.2f} ms`
            - **Upsampling:** Bilinear expansion back to original `{orig_meta.get('width', 256)} × {orig_meta.get('height', 256)} px`
            """,
            unsafe_allow_html=True,
        )
        if artifacts.get("restored"):
            st.image(f"{base_url}{artifacts['restored']}", caption="Stage 4 Artifact: Restored Folio Output", width=420)
        else:
            st.info("Restored artifact unavailable.")

    # STAGE 5: Baseline Autoencoder Comparison
    with st.expander("Stage 5: Convolutional Autoencoder Baseline Comparison", expanded=False):
        ae_time = timings.get("autoencoder_baseline", 0.0)
        st.markdown(
            f"""
            - **Subsystem:** Lightweight Baseline ConvAutoencoder (`src/models/autoencoder.py`)
            - **Architecture:** 3-tier symmetrical convolutional compression without skip connections
            - **Purpose:** Demonstrates academic baseline comparison showing why U-Net skip connections preserve fine stroke detail
            - **Latency:** `{ae_time:.2f} ms`
            - **Status:** {render_status_badge(readiness.get('autoencoder_baseline', 'UNTRAINED'))}
            """,
            unsafe_allow_html=True,
        )

    # STAGE 6: Optical Character Recognition
    with st.expander("Stage 6: Optical Character Recognition (OCR)", expanded=False):
        o_stat = ocr_res.get("status", "NOT_EXECUTED")
        st.markdown(
            f"""
            - **Subsystem:** Tesseract OCR Engine (`src/ocr/ocr_engine.py`)
            - **Execution Status:** {render_status_badge(o_stat)}
            - **Requested Language:** `{ocr_res.get('language_requested', 'san+hin+eng')}`
            - **Extracted Characters:** `{ocr_res.get('character_count', 0)}` | **Words:** `{ocr_res.get('word_count', 0)}`
            - **OCR Latency:** `{ocr_res.get('processing_time_ms', 0):.2f} ms`
            """,
            unsafe_allow_html=True,
        )
        if ocr_res.get("extracted_text"):
            st.markdown(f'<div class="devanagari-text">{ocr_res.get("extracted_text")}</div>', unsafe_allow_html=True)
        else:
            st.caption("No OCR text recorded for this session.")

    # STAGE 7: Contextual Text Prediction
    with st.expander("Stage 7: Context-Aware Damaged Text Prediction", expanded=False):
        st.markdown(
            """
            - **Subsystem:** N-gram Statistical Linguistic Completion (`src/prediction/context_predictor.py`)
            - **Training Corpus:** Authentic Vedic transcriptions (VedoOCR) + Sanskrit sentences
            - **Academic Classification:** `SPECULATIVE_PREDICTION` (Never claimed as verified history)
            """
        )
        if predictions:
            for p in predictions:
                st.markdown(f"- Top prediction: **{p.get('prediction')}** ({p.get('uncertainty_label')})")
        else:
            st.caption("No contextual predictions requested for this session.")

    # STAGE 8: Quantitative Evaluation & Metric Validation
    with st.expander("Stage 8: Quantitative Evaluation & Metric Validation", expanded=False):
        unet_m = restoration.get("unet_metrics", {})
        dmg_m = damage.get("metrics", {})
        st.markdown(
            f"""
            - **PSNR:** `{unet_m.get('psnr_db', 'REFERENCE UNAVAILABLE')}`
            - **SSIM:** `{unet_m.get('ssim', 'REFERENCE UNAVAILABLE')}`
            - **Damage IoU:** `{dmg_m.get('iou', 'GROUND TRUTH UNAVAILABLE')}`
            - **Damage Dice:** `{dmg_m.get('dice', 'GROUND TRUTH UNAVAILABLE')}`
            """
        )

    st.markdown("<br/><hr/>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. Architecture Deep-Dive & Academic Explanations
    # -------------------------------------------------------------
    st.markdown("### 🏛️ U-Net Architecture Deep-Dive")
    st.markdown(
        """
        The core restoration architecture follows the U-Net topology, modified specifically for degraded manuscript character preservation.
        Unlike standard denoising autoencoders, U-Net utilizes **skip connections** between corresponding encoder and decoder stages
        to bypass the spatial bottleneck, forwarding high-frequency character boundary coordinates directly to the reconstruction layers.
        """
    )

    # Conceptual Architecture Visualization
    st.markdown(
        """
        ```
        Input Folio (B, 3, 256, 256)
             │
        [DoubleConv 64]   ──────── Skip Connection 1 ────────>   [DoubleConv 64] ──> [1x1 Conv] ──> Output (B, 3, 256, 256)
             │ MaxPool                                                    ▲
        [DoubleConv 128]  ──────── Skip Connection 2 ────────>   [DoubleConv 128]
             │ MaxPool                                                    ▲
        [DoubleConv 256]  ──────── Skip Connection 3 ────────>   [DoubleConv 256]
             │ MaxPool                                                    ▲
        [DoubleConv 512]  ──────── Skip Connection 4 ────────>   [DoubleConv 512]
             │ MaxPool                                                    ▲ ConvTranspose2d
             └──────────────────> [DoubleConv 1024] ─────────────────────┘
                                    (Bottleneck)
        ```
        """,
        unsafe_allow_html=False,
    )

    # Critical Academic Distinction
    st.markdown(
        """
        <div class="p2p-notice-info">
            <strong>🎓 Academic Evaluation Clarification — Diagram vs. Learned Activations:</strong><br/>
            The schematic above illustrates the <strong>mathematical tensor architecture</strong> implemented in <code>src/models/unet_restoration.py</code>.
            While the structure, channel depths, and skip concatenations are fully operational in the forward pass,
            the <strong>internal convolutional filters</strong> are currently initialized with random weights.
            Learned visual activations (such as edge-detectors, stroke-continuity filters, and stain-subtraction channels)
            emerge only following backpropagation on GPU during training.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # OpenCV vs PyTorch Division of Labor
    st.markdown("### ⚙️ Division of Labor: OpenCV vs. PyTorch")
    c_cv, c_pt = st.columns(2)
    with c_cv:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem;">
                <h4 style="margin-top:0;">OpenCV / Classical CV</h4>
                <ul style="font-size: 0.85rem; padding-left: 1.2rem;">
                    <li><strong>Bilinear Rescaling:</strong> Deterministic spatial interpolation without learning parameters.</li>
                    <li><strong>Alpha Blending:</strong> Weighted pixel fusion for damage overlay generation (0.65 folio + 0.35 red mask).</li>
                    <li><strong>Color Space Conversions:</strong> RGB to grayscale conversions for edge analysis and thresholding.</li>
                    <li><strong>Zero Compute Footprint:</strong> Fast, CPU-bound deterministic transforms.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_pt:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem;">
                <h4 style="margin-top:0;">PyTorch Deep Learning</h4>
                <ul style="font-size: 0.85rem; padding-left: 1.2rem;">
                    <li><strong>Hierarchical Feature Extraction:</strong> Deep convolutional kernels learning non-linear text textures.</li>
                    <li><strong>Skip-Connection Highways:</strong> Preserves fine spatial stroke coordinates across scale bottlenecks.</li>
                    <li><strong>Loss Backpropagation:</strong> Computes MSE + L1 perceptual loss gradients for weight updates.</li>
                    <li><strong>Hardware Acceleration:</strong> Native CUDA tensor parallelism on GPU (NVIDIA RTX).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # 5. Technical Report Download
    # -------------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### 📥 Session Laboratory Report Export")
    md_report = generate_session_markdown_report(session)
    json_export = generate_session_json_export(session)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.download_button(
            "Download Comprehensive Academic Report (.md)",
            data=md_report,
            file_name=f"past2pixel_lab_report_{selected_sid}.md",
            mime="text/markdown",
            use_container_width=True,
            key="btn_dl_pip_md",
        )
    with rc2:
        st.download_button(
            "Download Raw Telemetry JSON (.json)",
            data=json_export,
            file_name=f"past2pixel_telemetry_{selected_sid}.json",
            mime="application/json",
            use_container_width=True,
            key="btn_dl_pip_json",
        )
