"""
Academic Technical Report Generator for Past2Pixel Restoration Sessions.
Compiles recorded metrics, device latencies, damage analyses, and text transcriptions
into reproducible Markdown and structured JSON reports. Strictly adheres to academic integrity.
"""
import json
from typing import Dict, Any


def generate_session_markdown_report(session: Dict[str, Any]) -> str:
    """
    Compiles an honest, academically defensible markdown report for a given session.
    Never claims trained accuracy or fabricated metrics when reference data is absent.
    """
    sid = session.get("session_id", "Unknown")
    created = session.get("created_at", "Unknown")
    owner = session.get("user_id", "Anonymous")
    status = session.get("status", "Unknown")
    orig_meta = session.get("original_metadata", {})
    rest_res = session.get("restoration_result") or {}
    ocr_res = session.get("ocr_result") or {}
    predictions = session.get("text_predictions") or []
    pers_info = session.get("persistence_info") or {}

    # Extract timings and readiness
    timings = rest_res.get("timings_ms", {})
    readiness = rest_res.get("model_readiness", {})
    damage = rest_res.get("damage_detection", {})
    restoration = rest_res.get("restoration", {})
    dev_info = rest_res.get("device_info", {})

    report_lines = [
        f"# Past2Pixel — Manuscript Restoration Technical Laboratory Report",
        f"**Session Identifier:** `{sid}`  ",
        f"**Execution Timestamp:** `{created}`  ",
        f"**User Account:** `{owner}`  ",
        f"**Final Session Status:** `{status}`  ",
        f"**Persistence Storage Mode:** `{pers_info.get('storage_mode', 'local_file_persistence')}`  ",
        "",
        "---",
        "",
        "## 1. Document Folio Input Specifications",
        f"- **Original File:** `{orig_meta.get('original_filename', 'folio.png')}`",
        f"- **Native Resolution:** `{orig_meta.get('width', 'N/A')} × {orig_meta.get('height', 'N/A')} px`",
        f"- **Color Mode / Encoding:** `{orig_meta.get('mode', 'RGB')} / {orig_meta.get('format', 'PNG')}`",
        f"- **Payload Size:** `{orig_meta.get('file_size_bytes', 0) / 1024:.2f} KB`",
        "",
        "---",
        "",
        "## 2. Deep Learning Pipeline & Model Readiness",
        "| Subsystem Module | Architecture | Checkpoint State | Academic Scientific Status |",
        "|---|---|---|---|",
        f"| Damage Localization | U-Net (Binary Segmentation) | `{readiness.get('damage_detector') or readiness.get('unet_damage_detector', 'UNKNOWN')}` | {'Trained Weights' if (readiness.get('damage_detector') or readiness.get('unet_damage_detector')) == 'READY_FOR_INFERENCE' else 'Untrained Initial Weights (Diagnostic)'} |",
        f"| Image Restoration | U-Net (Residual Skip Connections) | `{readiness.get('unet_restoration', 'UNKNOWN')}` | {'Trained Weights' if readiness.get('unet_restoration') == 'READY_FOR_INFERENCE' else 'Untrained Initial Weights (Diagnostic)'} |",
        f"| Baseline Comparison | Convolutional Autoencoder | `{readiness.get('autoencoder_baseline', 'UNKNOWN')}` | {'Trained Weights' if readiness.get('autoencoder_baseline') == 'READY_FOR_INFERENCE' else 'Untrained Initial Weights (Diagnostic)'} |",
        "",
        "> **Academic Disclosure**: If models are listed as `UNTRAINED`, outputs reflect the architectural pipeline flow and forward tensor dimensions, but must not be evaluated as finalized neural restorations.",
        "",
        "---",
        "",
        "## 3. Quantitative Damage Localization & Metrics",
        f"- **Damaged Surface Area:** `{damage.get('damaged_pixel_percentage', 'N/A')}%`",
        f"- **Damage Binary Threshold:** `{damage.get('threshold_used', 0.35)}`",
    ]

    # Damage metrics
    damage_metrics = damage.get("evaluation") or damage.get("metrics", {})
    if damage_metrics.get("metrics_available") or damage_metrics.get("status") == "CALCULATED":
        report_lines.append(f"- **Intersection-over-Union (IoU):** `{damage_metrics.get('iou', 0):.4f}`")
        report_lines.append(f"- **Dice Similarity Coefficient:** `{damage_metrics.get('dice', 0):.4f}`")
    else:
        report_lines.append(f"- **Segmentation Ground Truth:** `GROUND TRUTH MASK UNAVAILABLE (Calculated only on annotated test sets)`")

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Image Restoration Evaluation (PSNR / SSIM)",
    ])

    unet_metrics = restoration.get("unet_restoration_metrics") or restoration.get("unet_metrics", {})
    if unet_metrics.get("reference_available") and unet_metrics.get("status") == "CALCULATED":
        report_lines.append(f"- **U-Net PSNR:** `{unet_metrics.get('psnr_db', 0):.2f} dB`")
        report_lines.append(f"- **U-Net SSIM:** `{unet_metrics.get('ssim', 0):.4f}`")
    else:
        report_lines.append("- **Restoration Quality Metrics:** `REFERENCE UNAVAILABLE (PSNR/SSIM require a clean ground-truth reference image)`")

    report_lines.extend([
        "",
        "---",
        "",
        "## 5. Optical Character Recognition (OCR)",
        f"- **OCR Status:** `{ocr_res.get('status', 'NOT_EXECUTED')}`",
        f"- **Engine & Language Pack:** `{ocr_res.get('language_requested', 'san+hin+eng')}`",
        f"- **Extracted Character Count:** `{ocr_res.get('character_count', 0)}`",
        f"- **Authentic Engine Extraction:** `{ocr_res.get('is_authentic_ocr', False)}`",
        f"- **OCR Processing Latency:** `{ocr_res.get('processing_time_ms', 0):.2f} ms`",
        "",
        "### Extracted Text Content:",
        "```text",
        ocr_res.get("extracted_text") or "(No text extracted or OCR engine unavailable)",
        "```",
        "",
        "---",
        "",
        "## 6. Context-Aware Damaged Text Predictions",
    ])

    if predictions:
        for idx, pred in enumerate(predictions, 1):
            report_lines.append(f"**Prediction #{idx}:** Top candidate `{pred.get('prediction', 'N/A')}`")
            report_lines.append(f"- Uncertainty Indicator: `{pred.get('uncertainty_label', 'SPECULATIVE')}`")
            report_lines.append(f"- Context: `... {pred.get('surrounding_context', {}).get('left', '')} [?] {pred.get('surrounding_context', {}).get('right', '')} ...`")
            cands = pred.get("candidates", [])
            if cands:
                cand_str = ", ".join([f"{c.get('word', c.get('text', ''))} (score: {c.get('score', c.get('context_score', 0)):.3f})" for c in cands])
                report_lines.append(f"- Candidate Rank: {cand_str}")
    else:
        report_lines.append("*(No contextual text predictions were recorded for this session)*")

    report_lines.extend([
        "",
        "---",
        "",
        "## 7. Execution Latencies & Compute Profile",
        f"- **Host Device:** `{dev_info.get('device_name', 'CPU')} ({dev_info.get('device_type', 'cpu').upper()})`",
        f"- **Total Pipeline Latency:** `{timings.get('total_pipeline', 0):.2f} ms`",
        f"  - Preprocessing: `{timings.get('preprocessing', 0):.2f} ms`",
        f"  - Damage Detection: `{timings.get('damage_detection', 0):.2f} ms`",
        f"  - U-Net Restoration: `{timings.get('unet_restoration', 0):.2f} ms`",
        f"  - Baseline Autoencoder: `{timings.get('autoencoder_baseline', 0):.2f} ms`",
        f"  - Postprocessing / Artifact Save: `{timings.get('postprocessing_and_saving', 0):.2f} ms`",
        "",
        "---",
        "*Report generated automatically by Past2Pixel Research System. Designed for Faculty & Peer Review Evaluation.*",
    ])

    return "\n".join(report_lines)


def generate_session_json_export(session: Dict[str, Any]) -> str:
    """Exports entire verified session document as pretty-printed JSON."""
    return json.dumps(session, indent=2, default=str)
