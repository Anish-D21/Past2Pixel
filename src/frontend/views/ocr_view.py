"""
Optical Character Recognition (OCR) & Text Extraction View for Past2Pixel.
Provides authentic Tesseract extraction, Devanagari/Sanskrit Unicode display,
CER/WER error rate evaluation against ground-truth references, and contextual text prediction.
"""
from PIL import Image
import io
import streamlit as st
from ..api_client import api_client, ApiException
from ..state import get_token, get_active_session_id
from ..components.styles import render_status_badge
from ...prediction.context_predictor import calculate_cer, calculate_wer


def render_ocr_view():
    st.markdown('<h1 class="manuscript-title">Optical Character Recognition & Preservation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="manuscript-subtitle">Extract authentic Devanagari and Vedic Sanskrit text from ancient folios with scientific integrity.</p>',
        unsafe_allow_html=True,
    )

    token = get_token()
    active_sid = get_active_session_id()

    # -------------------------------------------------------------
    # 1. OCR Engine Telemetry & Availability Diagnostic
    # -------------------------------------------------------------
    ocr_diag = {}
    is_installed = False
    try:
        readiness = api_client.get_readiness()
        ocr_diag = readiness.get("ocr", {})
        is_installed = ocr_diag.get("is_installed", False)
    except Exception:
        is_installed = False

    if not is_installed:
        st.markdown(
            f"""
            <div class="p2p-notice-warning">
                <strong>⚠️ Tesseract OCR Engine Status: UNAVAILABLE</strong><br/>
                {ocr_diag.get('reason', 'Tesseract executable was not detected on Windows system PATH or standard directories.')}<br/>
                In adherence to academic integrity, Past2Pixel <strong>never fabricates OCR output</strong> when the underlying engine is offline.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🛠️ Windows Tesseract & Language Pack Installation Instructions", expanded=True):
            setup = ocr_diag.get("setup_instructions") or {}
            for step in setup.get("steps", []):
                st.markdown(f"- {step}")
            st.markdown(f"**Required Language Packs**: {', '.join(setup.get('required_language_packs', ['san', 'hin']))}")
            st.markdown(f"**Official UB-Mannheim Windows Binary**: [Download Installer]({setup.get('download_url', 'https://github.com/UB-Mannheim/tesseract/wiki')})")

    # -------------------------------------------------------------
    # 2. Extraction Mode Selection
    # -------------------------------------------------------------
    tab_session, tab_direct, tab_predict = st.tabs([
        "📜 Run OCR on Existing Session",
        "📤 Direct Image OCR",
        "🔮 Contextual Text Prediction",
    ])

    # -------------------------------------------------------------
    # TAB 1: Session-Based OCR
    # -------------------------------------------------------------
    with tab_session:
        st.markdown("### Process Existing Manuscript Session")
        sid_input = st.text_input(
            "Session Identifier",
            value=active_sid or "",
            placeholder="e.g. p2p_xxxxxxxxxxxx_1792515000",
            help="Enter a session ID from Restore Manuscript or Restoration History.",
        )
        lang_choice = st.selectbox(
            "OCR Language Pack Strategy",
            options=["san+hin+eng", "san", "hin", "eng"],
            index=0,
            key="ocr_session_lang",
            help="'san' = Vedic/Classical Sanskrit, 'hin' = Hindi Devanagari, 'eng' = English catalog numbers",
        )

        if st.button("Execute Session OCR", type="primary", key="btn_session_ocr"):
            if not sid_input.strip():
                st.error("Please enter or select a valid session ID.")
            else:
                with st.spinner("Extracting text via Tesseract OCR..."):
                    try:
                        ocr_res = api_client.run_ocr(session_id=sid_input.strip(), language=lang_choice, token=token)
                        st.session_state["last_ocr_result"] = ocr_res
                        st.success("OCR extraction completed!")
                    except ApiException as e:
                        st.error(f"OCR execution failed: {e.detail}")

    # -------------------------------------------------------------
    # TAB 2: Direct Image OCR
    # -------------------------------------------------------------
    with tab_direct:
        st.markdown("### Direct Manuscript Folio Upload")
        direct_file = st.file_uploader(
            "Upload image for instant OCR",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            key="ocr_direct_upload",
        )
        direct_lang = st.selectbox(
            "Target Language Pack",
            options=["san+hin+eng", "san", "hin", "eng"],
            index=0,
            key="ocr_direct_lang",
        )

        if st.button("Extract Text Directly", type="primary", key="btn_direct_ocr"):
            if direct_file is None:
                st.error("Please choose a manuscript image file to upload.")
            else:
                with st.spinner("Processing image through OCR engine..."):
                    try:
                        ocr_res = api_client.run_direct_ocr(
                            file_bytes=direct_file.getvalue(),
                            filename=direct_file.name,
                            language=direct_lang,
                            token=token,
                        )
                        st.session_state["last_ocr_result"] = ocr_res
                        st.success("Direct text extraction completed!")
                    except ApiException as e:
                        st.error(f"Direct OCR failed: {e.detail}")

    # -------------------------------------------------------------
    # TAB 3: Contextual Damaged Text Prediction
    # -------------------------------------------------------------
    with tab_predict:
        st.markdown("### Context-Aware Damaged Text Prediction")
        st.caption("Statistically estimates missing Sanskrit / Devanagari words from surrounding legible context.")

        st.markdown(
            """
            <div class="p2p-notice-info">
                <strong>Academic Defensibility Rule:</strong> Linguistic predictions are derived from n-gram corpus statistics
                of Vedic manuscript transcriptions (VedoOCR) and Sanskrit sentences. All predictions are explicitly labeled
                <code>is_speculative=True</code> and must never be conflated with authentic OCR text.
            </div>
            """,
            unsafe_allow_html=True,
        )

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            left_ctx = st.text_input("Left Context (Preceding words)", value="धर्मक्षेत्रे कुरुक्षेत्रे समवेता", placeholder="e.g. धर्मक्षेत्रे कुरुक्षेत्रे समवेता")
        with pcol2:
            right_ctx = st.text_input("Right Context (Succeeding words)", value="मामकाः पाण्डवाश्चैव", placeholder="e.g. मामकाः पाण्डवाश्चैव")

        k_cand = st.slider("Number of Candidates", min_value=1, max_value=5, value=3)

        if st.button("Predict Missing Span", type="primary", key="btn_predict_text"):
            if not left_ctx.strip() and not right_ctx.strip():
                st.error("Please provide at least one surrounding context span.")
            else:
                with st.spinner("Querying contextual language model..."):
                    try:
                        pred_res = api_client.predict_text(
                            left_context=left_ctx,
                            right_context=right_ctx,
                            max_candidates=k_cand,
                            token=token,
                        )
                        st.session_state["last_pred_result"] = pred_res
                    except ApiException as e:
                        st.error(f"Prediction failed: {e.detail}")

        pred_data = st.session_state.get("last_pred_result")
        if pred_data:
            st.markdown("#### Prediction Output")
            top_word = pred_data.get("prediction") or "(No high-probability candidate)"
            st.markdown(
                f"""
                <div style="background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #7e22ce; text-transform: uppercase;">
                        {pred_data.get('uncertainty_label', 'SPECULATIVE_PREDICTION')}
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #581c87; margin: 0.5rem 0;">
                        {top_word}
                    </div>
                    <div style="font-size: 0.85rem; color: #6b21a8;">
                        Reconstructed Sentence: <em>{left_ctx} <strong>[{top_word}]</strong> {right_ctx}</em>
                    </div>
                    <div style="font-size: 0.75rem; color: #9333ea; margin-top: 0.5rem;">
                        ⚠️ {pred_data.get('disclaimer')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Ranked candidates table
            candidates = pred_data.get("candidates", [])
            if candidates:
                st.markdown("**Alternative Candidates:**")
                for c in candidates:
                    word = c.get("word") or c.get("text")
                    score = c.get("score") or c.get("context_score", 0.0)
                    st.markdown(f"- **{word}** — likelihood score: `{score:.4f}`")

    # -------------------------------------------------------------
    # 3. Display Last OCR Result & CER/WER Evaluation
    # -------------------------------------------------------------
    ocr_result = st.session_state.get("last_ocr_result")
    if ocr_result:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### Extracted Text & Scientific Verification")

        o_stat = ocr_result.get("status", "UNKNOWN")
        is_auth = ocr_result.get("is_authentic_ocr", False)
        char_cnt = ocr_result.get("character_count", 0)
        word_cnt = ocr_result.get("word_count", 0)
        time_ms = ocr_result.get("processing_time_ms", 0.0)

        # Telemetry row
        om1, om2, om3, om4 = st.columns(4)
        with om1:
            st.metric("OCR Engine Status", o_stat)
        with om2:
            st.metric("Execution Latency", f"{time_ms:.1f} ms")
        with om3:
            st.metric("Extracted Characters", char_cnt)
        with om4:
            st.metric("Authentic Engine", "Yes" if is_auth else "No / Unavailable")

        # Display extracted text in styled Devanagari container
        extracted_text = ocr_result.get("extracted_text", "").strip()
        if extracted_text:
            st.markdown("#### Extracted Unicode Text:")
            st.markdown(f'<div class="devanagari-text">{extracted_text}</div>', unsafe_allow_html=True)
        else:
            st.info("No Unicode text was returned by the OCR engine for this image.")

        # ---------------------------------------------------------
        # CER / WER Evaluation Module (Academic Differentiator)
        # ---------------------------------------------------------
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### Quantitative Error Evaluation (CER / WER)")
        st.caption("Evaluate character-level and word-level error rates against ground-truth manuscript transcriptions.")

        ground_truth_input = st.text_area(
            "Ground-Truth Reference Transcription (Optional)",
            placeholder="Paste verified transcript (e.g. from VedoOCR master.tsv or scholarly catalog) to compute CER & WER...",
            help="If no ground truth is provided, error metrics remain strictly unavailable.",
        )

        if ground_truth_input.strip() and extracted_text:
            cer_val = calculate_cer(reference=ground_truth_input.strip(), hypothesis=extracted_text)
            wer_val = calculate_wer(reference=ground_truth_input.strip(), hypothesis=extracted_text)

            em1, em2 = st.columns(2)
            with em1:
                st.metric("Character Error Rate (CER)", f"{cer_val * 100:.2f}%")
            with em2:
                st.metric("Word Error Rate (WER)", f"{wer_val * 100:.2f}%")
        else:
            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #64748b; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem;">
                    ℹ️ <strong>Evaluation Metrics:</strong> <code>CER / WER UNAVAILABLE</code> — Character and Word Error Rates are only calculated when an authentic ground-truth reference is provided.
                </div>
                """,
                unsafe_allow_html=True,
            )
