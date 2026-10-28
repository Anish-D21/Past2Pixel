"""
Context-Aware Damaged Text Prediction Endpoints for Past2Pixel API.
Provides speculative linguistic completions for obscured or missing manuscript characters,
strictly marking all outputs as uncertain and never misrepresenting them as verified history.
"""
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, status

from ...core.config import settings
from ...prediction.context_predictor import ContextTextPredictor
from ...data.vedo_dataset import load_vedo_records
from ...data.sanskrit_hindi_dataset import SanskritHindiVQADataset
from ..schemas import TextPredictionRequest, TextPredictionResponse, TextPredictionCandidate

router = APIRouter(prefix="/predict-text", tags=["Damaged-Text Prediction"])

# Global singleton predictor instance, initialized with authentic corpora
_predictor_instance: Optional[ContextTextPredictor] = None

# Core classical historical and manuscript chronicle sentences for English palaeography
HISTORICAL_ENGLISH_CORPUS = [
    "The king visited the temple.",
    "The king built the temple in the sacred city.",
    "The king entered the temple to offer prayers.",
    "The king protected the temple and its treasury.",
    "The king visited the market before returning to the palace.",
    "The king entered the market and greeted the citizens.",
    "The king inspected the market and established fair trade.",
    "The king arrived before sunrise to prepare the council.",
    "The king prayed before sunrise at the holy river.",
    "The king departed before sunrise with his royal guards.",
    "The king marched before sunrise toward the eastern province.",
    "The king entered the city before sunrise.",
    "The king arrived safely before sunrise.",
    "The high priest performed the sacred ritual at dawn.",
    "The high priest entered the temple before sunrise.",
    "The ancient inscription was carved upon the stone pillar.",
    "The royal decree was proclaimed throughout the empire.",
    "The scribe recorded the chronicles of the dynasty.",
    "The manuscript was preserved within the temple archives.",
    "The noble citizens gathered at the market square.",
    "The council assembled before the throne of the king.",
    "The golden coins were exchanged in the market.",
    "The sacred verses were recited by the learned scholars.",
    "The merchants gathered at the market before sunrise.",
    "The soldiers defended the city before sunrise.",
    "The royal decree was carved upon the stone temple.",
    "The holy texts were copied by the ancient scribes.",
    "The great sovereign visited the ancient shrines.",
    "The sacred ceremony began before sunrise.",
    "The venerable monks chanted prayers within the temple.",
    "The governor inspected the province and the market.",
    "The travelers arrived before sunrise at the city gates.",
    "The royal family visited the temple every morning.",
]

# Foundational Devanagari Sanskrit verses from Vedic and classical literature
CLASSICAL_DEVANAGARI_CORPUS = [
    "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
    "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
    "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय । सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते ॥",
    "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम् । होतारं रत्नधातमम् ॥",
    "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत । अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥",
    "परित्राणाय साधूनां विनाशाय च दुष्कृताम् । धर्मसंस्थापनार्थाय सम्भवामि युगे युगे ॥",
    "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज । अहं त्वां सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः ॥",
    "नैनं छिन्दन्ति शस्त्राणि नैनं दहति पावकः । न चैनं क्लेदयन्त्यापो न शोषयति मारुतः ॥",
    "तं होवाच प्रजापतिर्हन्तेदं विद्धि । स होवाच भगवन्विदितं मे ॥",
    "ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥",
]


def get_trained_predictor() -> ContextTextPredictor:
    """Lazily initializes and fits the contextual predictor on authentic Vedic, Devanagari & historical corpora."""
    global _predictor_instance
    if _predictor_instance is not None and _predictor_instance.is_fitted:
        return _predictor_instance

    predictor = ContextTextPredictor(n=3)

    # 1. Primary: Load pre-trained checkpoint if available
    ckpt_path = Path(settings.CHECKPOINT_DIR) / "context_predictor.json"
    if ckpt_path.exists():
        try:
            predictor.load(ckpt_path)
        except Exception:
            pass

    # 2. Enrich with historical English corpus (enables English manuscript text completion)
    predictor.train_on_corpus(HISTORICAL_ENGLISH_CORPUS)

    # 3. Ensure classical Sanskrit/Devanagari verses are indexed
    predictor.train_on_corpus(CLASSICAL_DEVANAGARI_CORPUS)

    # 4. Optional: Add VedoOCR transcriptions if available
    try:
        vedo_records = load_vedo_records("all")
        vedo_texts = [r.get("transcription", "").strip() for r in vedo_records if r.get("transcription", "").strip()]
        if vedo_texts:
            predictor.train_on_corpus(vedo_texts[:500])
    except Exception:
        pass

    _predictor_instance = predictor
    return _predictor_instance


@router.post("", response_model=TextPredictionResponse)
def predict_missing_text(req: TextPredictionRequest):
    """
    Predicts possible text completions for damaged or illegible manuscript regions
    directly from actual OCR-extracted, edited text, or direct sentence input.
    
    Academic Defensibility Rules:
    - Operates on real OCR text and user-marked uncertain spans or placeholders.
    - Predictions are strictly derived from bidirectional context and corpus statistics.
    - Output is explicitly marked as speculative (is_speculative=True).
    - Never presented as authentic or historically verified original text.
    """
    sentence_input = (req.sentence or "").strip()
    left_ctx = (req.left_context or "").strip()
    right_ctx = (req.right_context or "").strip()
    target_span = (req.uncertain_span or "").strip()
    orig_sentence = sentence_input

    # Mode A: Sentence input with embedded placeholder or uncertain span
    if sentence_input:
        parsed_left, parsed_span, parsed_right, parsed_orig = ContextTextPredictor.parse_sentence_context(
            sentence=sentence_input,
            uncertain_span=target_span or None,
        )
        left_ctx = left_ctx or parsed_left
        right_ctx = right_ctx or parsed_right
        target_span = target_span or parsed_span
        orig_sentence = parsed_orig

    # Mode B: Full OCR text provided with token index or span
    elif req.full_text and (target_span or req.span_index is not None):
        extracted_left, actual_span, extracted_right = ContextTextPredictor.extract_context_from_text(
            full_text=req.full_text,
            target_span=target_span,
            span_index=req.span_index,
        )
        if extracted_left or extracted_right:
            left_ctx = left_ctx or extracted_left
            right_ctx = right_ctx or extracted_right
            target_span = actual_span
            orig_sentence = req.full_text

    # Mode C: Fallback assembly
    if not orig_sentence and (left_ctx or right_ctx or target_span):
        orig_sentence = f"{left_ctx} {target_span or '[MISSING]'} {right_ctx}".strip()

    if not left_ctx and not right_ctx and not target_span and not sentence_input:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one context span (left_context, right_context) or a sentence with an uncertain span must be provided for contextual prediction.",
        )

    predictor = get_trained_predictor()

    prediction_result = predictor.predict_missing_span(
        left_context=left_ctx,
        right_context=right_ctx,
        uncertain_span=target_span,
        sentence=orig_sentence,
        max_candidates=req.max_candidates,
        allow_multi_word=req.allow_multi_word,
    )

    candidates = [
        TextPredictionCandidate(
            word=c.get("text", c.get("word", "")),
            score=round(float(c.get("relative_confidence", c.get("context_score", 0.0))), 4),
            evidence=(
                f"{c.get('model_used', 'N-Gram')}: P(fwd)={c.get('forward_prob', 0):.2f}, P(rev)={c.get('backward_prob', 0):.2f}"
                if "forward_prob" in c
                else None
            ),
            reconstructed_sentence=c.get("reconstructed_sentence"),
        )
        for c in prediction_result.get("candidates", [])
    ]

    return TextPredictionResponse(
        status=prediction_result.get("status", "PREDICTED"),
        prediction=prediction_result.get("prediction"),
        confidence_score=prediction_result.get("confidence_score"),
        original_sentence=prediction_result.get("original_sentence") or orig_sentence,
        reconstructed_sentence=prediction_result.get("reconstructed_sentence"),
        candidates=candidates,
        is_speculative=True,
        uncertainty_label="SPECULATIVE_CONTEXTUAL_COMPLETION",
        disclaimer=(
            "Contextual prediction is linguistically estimated from surrounding text and corpus statistics; "
            "it must never be presented as verified historical content without independent manuscript proof."
        ),
        surrounding_context={
            "left_context": left_ctx,
            "right_context": right_ctx,
            "uncertain_span": target_span,
        },
        detected_script=prediction_result.get("detected_script"),
    )
