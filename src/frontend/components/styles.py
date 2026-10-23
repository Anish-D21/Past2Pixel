"""
Academic Research UI Styling and Status Badge Rendering for Past2Pixel.
Provides clean CSS variables, elegant typography, and distinct visual indicators
for READY, UNTRAINED, UNAVAILABLE, and REFERENCE UNAVAILABLE states.
"""
import streamlit as st

CUSTOM_CSS = """
<style>
/* -------------------------------------------------------------
   Typography & Base Layout
   ------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Academic Title Typography */
.manuscript-title {
    font-family: 'Cinzel', Georgia, serif;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #1a202c;
    margin-bottom: 0.25rem;
}

.manuscript-subtitle {
    color: #4a5568;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

/* -------------------------------------------------------------
   Status Pill Badges
   ------------------------------------------------------------- */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-radius: 9999px;
    vertical-align: middle;
}

.badge-ready {
    background-color: #def7ec;
    color: #03543f;
    border: 1px solid #bcf0da;
}

.badge-untrained {
    background-color: #fef08a;
    color: #854d0e;
    border: 1px solid #fde047;
}

.badge-unavailable {
    background-color: #f3f4f6;
    color: #4b5563;
    border: 1px solid #e5e7eb;
}

.badge-ref-unavailable {
    background-color: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
}

.badge-speculative {
    background-color: #f3e8ff;
    color: #6b21a8;
    border: 1px solid #e9d5ff;
}

/* -------------------------------------------------------------
   Academic Metric & Callout Cards
   ------------------------------------------------------------- */
.p2p-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}

.p2p-metric-card {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}

.p2p-metric-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #111827;
    margin: 0.25rem 0;
    font-family: 'JetBrains Mono', monospace;
}

.p2p-metric-lbl {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6b7280;
    font-weight: 600;
}

/* Notices and Academic Callouts */
.p2p-notice-warning {
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 1rem 1.25rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 1.25rem;
    color: #92400e;
}

.p2p-notice-info {
    background-color: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 1rem 1.25rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 1.25rem;
    color: #1e40af;
}

/* Code block styling */
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Devanagari text presentation */
.devanagari-text {
    font-size: 1.35rem;
    line-height: 1.8;
    color: #1a202c;
    background: #fdfaf6;
    padding: 1.2rem;
    border-radius: 6px;
    border: 1px solid #eedcce;
}
</style>
"""


def apply_custom_styles() -> None:
    """Injects custom academic stylesheet into the active Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_status_badge(status_text: str) -> str:
    """Returns styled HTML badge span for a given status keyword."""
    s = (status_text or "").upper().strip()
    if "READY" in s or "CONNECTED" in s or "SUCCESS" in s or "OK" in s or "SAVED" in s:
        css_class = "badge-ready"
    elif "UNTRAINED" in s or "DEGRADED" in s or "RANDOM" in s:
        css_class = "badge-untrained"
    elif "SPECULATIVE" in s or "PREDICTED" in s:
        css_class = "badge-speculative"
    elif "REFERENCE" in s or "GROUND TRUTH" in s:
        css_class = "badge-ref-unavailable"
    else:
        css_class = "badge-unavailable"

    return f'<span class="badge {css_class}">{s}</span>'
