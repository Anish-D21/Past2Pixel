"""
Context-Aware Damaged Text Prediction Endpoints for Past2Pixel API.
Provides speculative linguistic completions for obscured or missing manuscript characters,
strictly marking all outputs as uncertain and never misrepresenting them as verified history.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, status

from ...core.config import settings
from ...prediction.context_predictor import ContextTextPredictor
from ...data.vedo_dataset import load_vedo_records
from ...data.sanskrit_hindi_dataset import SanskritHindiVQADataset
from ..schemas import TextPredictionRequest, TextPredictionResponse, TextPredictionCandidate

router = APIRouter(prefix="/predict-text", tags=["Damaged-Text Prediction"])

# Global singleton predictor instance, initialized with authentic corpora
_predictor_instance: Optional[ContextTextPredictor] = None


def get_trained_predictor() -> ContextTextPredictor:
    """Lazily initializes and fits the contextual predictor on authentic Vedic & Devanagari corpora."""
    global _predictor_instance
    if _predictor_instance is None or not _predictor_instance.is_fitted:
        predictor = ContextTextPredictor(n=3)
        corpus = []

        # 1. Primary: Expert Vedic Sanskrit transcriptions from VedoOCR
        try:
            vedo_records = load_vedo_records("all")
            for r in vedo_records:
                txt = r.get("transcription", "").strip()
                if txt:
                    corpus.append(txt)
        except Exception:
            pass

        # 2. Secondary: Authentic Devanagari sentences from Sanskrit-Hindi OCR VQA
        try:
            vqa_ds = SanskritHindiVQADataset(max_preload_texts=3000)
            vqa_texts = vqa_ds.get_text_corpus(max_samples=3000)
            corpus.extend(vqa_texts)
        except Exception:
            pass

        if corpus:
            predictor.train_on_corpus(corpus)
        else:
            # Fallback corpus if raw datasets inaccessible
            predictor.train_on_corpus([
                "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
                "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
                "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय । सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते ॥",
            ])

        _predictor_instance = predictor

    return _predictor_instance


@router.post("", response_model=TextPredictionResponse)
def predict_missing_text(req: TextPredictionRequest):
    """
    Predicts possible text completions for damaged or illegible manuscript regions.
    
    Academic Defensibility Rules:
    - Predictions are strictly derived from surrounding context and corpus statistics.
    - Output is explicitly marked as speculative (is_speculative=True).
    - Never presented as authentic or historically verified original text.
    """
    if not req.left_context.strip() and not req.right_context.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one context span (left_context or right_context) must be provided for contextual prediction.",
        )

    predictor = get_trained_predictor()

    prediction_result = predictor.predict_missing_span(
        left_context=req.left_context.strip(),
        right_context=req.right_context.strip(),
        max_candidates=req.max_candidates,
    )

    candidates = [
        TextPredictionCandidate(
            word=c.get("text", c.get("word", "")),
            score=round(float(c.get("context_score", c.get("score", 0.0))), 4),
        )
        for c in prediction_result.get("candidates", [])
    ]

    return TextPredictionResponse(
        status=prediction_result.get("status", "PREDICTED"),
        prediction=prediction_result.get("prediction"),
        candidates=candidates,
        is_speculative=True,
        uncertainty_label="SPECULATIVE_CONTEXTUAL_COMPLETION",
        disclaimer=(
            "Contextual prediction is linguistically estimated from surrounding text and corpus statistics; "
            "it must never be presented as verified historical content without independent manuscript confirmation."
        ),
        surrounding_context={
            "left_context": req.left_context,
            "right_context": req.right_context,
        },
    )
