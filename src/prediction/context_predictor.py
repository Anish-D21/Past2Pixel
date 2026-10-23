"""
Context-Aware Damaged Text Prediction Engine.
Uses n-gram statistical language modeling and corpus likelihood to suggest plausible
text completions for damaged/unreadable manuscript regions.
Strictly separates recognized OCR characters from speculative predictions.
"""
from typing import Dict, Any, List, Optional, Tuple
import re
from collections import defaultdict, Counter
import json
from pathlib import Path

from ..core.config import settings


class ContextTextPredictor:
    """
    Predicts plausible completions for unreadable or damaged Sanskrit/Hindi text spans
    using surrounding linguistic context (n-gram word and character models).
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.word_ngrams: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.char_ngrams: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocabulary: Counter = Counter()
        self.total_words: int = 0
        self.is_fitted: bool = False

    def train_on_corpus(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Trains n-gram statistical language model on Devanagari text sentences.
        """
        word_count = 0
        for sentence in sentences:
            # Tokenize words preserving Devanagari characters
            words = [w for w in re.split(r"[\s,।॥]+", sentence) if w]
            word_count += len(words)
            for w in words:
                self.vocabulary[w] += 1

            # Word n-grams
            padded_words = ["<s>"] * (self.n - 1) + words + ["</s>"]
            for i in range(len(padded_words) - self.n + 1):
                ctx = tuple(padded_words[i : i + self.n - 1])
                target = padded_words[i + self.n - 1]
                self.word_ngrams[ctx][target] += 1

            # Character n-grams for subword recovery
            chars = list(sentence)
            padded_chars = ["^"] * 2 + chars + ["$"]
            for i in range(len(padded_chars) - 2):
                c_ctx = tuple(padded_chars[i : i + 2])
                c_tgt = padded_chars[i + 2]
                self.char_ngrams[c_ctx][c_tgt] += 1

        self.total_words = word_count
        self.is_fitted = True

        return {
            "sentences_trained": len(sentences),
            "total_words_processed": word_count,
            "unique_vocabulary_size": len(self.vocabulary),
            "unique_contexts": len(self.word_ngrams),
        }

    @staticmethod
    def extract_context_from_text(
        full_text: str,
        target_span: str,
        span_index: Optional[int] = None,
        context_window_words: int = 4,
    ) -> Tuple[str, str, str]:
        """
        Extracts surrounding left and right context words for an uncertain word/span
        directly from actual OCR-extracted or edited text.
        Returns: (left_context, actual_span, right_context)
        """
        import unicodedata
        normalized_full = unicodedata.normalize("NFC", full_text.strip())
        normalized_target = unicodedata.normalize("NFC", target_span.strip())

        words = [w for w in re.split(r"(\s+|[।॥,;]+)", normalized_full) if w.strip()]
        if not words:
            return "", target_span, ""

        # Locate word index
        target_idx = -1
        if span_index is not None and 0 <= span_index < len(words):
            target_idx = span_index
        else:
            # Search for matching word token
            for i, w in enumerate(words):
                if w == normalized_target or normalized_target in w or w in normalized_target:
                    target_idx = i
                    break

        if target_idx == -1:
            # Fallback: substring split if possible
            if normalized_target in normalized_full:
                parts = normalized_full.split(normalized_target, 1)
                left_part = parts[0].strip()
                right_part = parts[1].strip()
                left_w = [w for w in re.split(r"[\s,।॥]+", left_part) if w][-context_window_words:]
                right_w = [w for w in re.split(r"[\s,।॥]+", right_part) if w][:context_window_words]
                return " ".join(left_w), normalized_target, " ".join(right_w)
            return "", target_span, ""

        actual_span = words[target_idx]
        left_slice = words[max(0, target_idx - context_window_words) : target_idx]
        right_slice = words[target_idx + 1 : min(len(words), target_idx + 1 + context_window_words)]

        return " ".join(left_slice), actual_span, " ".join(right_slice)

    def predict_missing_span(
        self,
        left_context: str,
        right_context: str,
        uncertain_span: Optional[str] = None,
        max_candidates: int = 4,
        min_confidence: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Suggests candidate completions for a missing/damaged span between left_context and right_context,
        optionally constrained by surviving character fragments in uncertain_span.
        
        Returns structured prediction marked as SPECULATIVE with explicit uncertainty flags.
        """
        import unicodedata
        norm_left = unicodedata.normalize("NFC", left_context.strip())
        norm_right = unicodedata.normalize("NFC", right_context.strip())
        norm_span = unicodedata.normalize("NFC", (uncertain_span or "").strip())

        if not self.is_fitted:
            return {
                "status": "UNAVAILABLE",
                "reason": "Context predictor model is not yet fitted on textual corpus.",
                "candidates": [],
                "prediction": None,
                "is_speculative": True,
            }

        # 1. Left-context n-gram extraction
        left_words = [w for w in re.split(r"[\s,।॥]+", norm_left) if w]
        if not left_words:
            ctx = ("<s>",) if self.n == 2 else ("<s>", "<s>")
        elif len(left_words) == 1:
            ctx = ("<s>", left_words[-1]) if self.n == 3 else (left_words[-1],)
        else:
            ctx = tuple(left_words[-(self.n - 1) :])

        candidates_counter = self.word_ngrams.get(ctx, Counter())
        total_ctx = sum(candidates_counter.values()) if candidates_counter else 0

        # 2. Right-context word for bidirectional scoring
        right_words = [w for w in re.split(r"[\s,।॥]+", norm_right) if w]
        next_word = right_words[0] if right_words else None

        # Clean surviving characters in uncertain_span (strip placeholder markers like ?, [], ...)
        clean_prefix = re.sub(r"[\[\]\?\.\_\s\(\)]+", "", norm_span)

        # 3. Candidate generation pool
        candidate_pool = set()
        for w, _ in candidates_counter.most_common(20):
            if w not in ["<s>", "</s>"]:
                candidate_pool.add(w)

        # Also add unigram backoff candidates if pool is small
        if len(candidate_pool) < max_candidates * 2:
            for w, _ in self.vocabulary.most_common(max_candidates * 4):
                if w not in ["<s>", "</s>"]:
                    candidate_pool.add(w)

        # If prefix/fragment exists, also include vocabulary words starting with fragment
        if clean_prefix and len(clean_prefix) >= 2:
            for w in self.vocabulary:
                if w.startswith(clean_prefix) and w not in ["<s>", "</s>"]:
                    candidate_pool.add(w)

        if not candidate_pool:
            return {
                "status": "INSUFFICIENT_CONTEXT",
                "reason": "Context is too sparse to provide a mathematically defensible prediction.",
                "candidates": [],
                "prediction": None,
                "is_speculative": True,
            }

        # 4. Bidirectional scoring for each candidate
        total_vocab_occ = max(sum(self.vocabulary.values()), 1)
        scored_candidates = []

        for cand in candidate_pool:
            # A. Left-context transition probability
            p_left = (candidates_counter.get(cand, 0) / total_ctx) if total_ctx > 0 else 0.0
            p_unigram = self.vocabulary.get(cand, 0) / total_vocab_occ

            # Combined forward probability (Jelinek-Mercer style interpolation)
            forward_prob = 0.75 * p_left + 0.25 * p_unigram

            # B. Right-context subsequent transition probability P(next_word | cand)
            backward_prob = 0.0
            if next_word:
                cand_subsequent = self.word_ngrams.get((cand,), Counter())
                cand_subsequent_bi = self.word_ngrams.get(("<s>", cand), Counter())
                sub_count = cand_subsequent.get(next_word, 0) + cand_subsequent_bi.get(next_word, 0)
                sub_total = sum(cand_subsequent.values()) + sum(cand_subsequent_bi.values())
                if sub_total > 0:
                    backward_prob = sub_count / sub_total

            # C. Partial fragment similarity bonus
            fragment_match = 1.0
            if clean_prefix:
                if cand == clean_prefix:
                    fragment_match = 2.0
                elif cand.startswith(clean_prefix):
                    fragment_match = 1.5
                elif clean_prefix in cand:
                    fragment_match = 1.2
                else:
                    fragment_match = 0.5  # Penalize if does not match visible letters

            # Composite bidirectional score
            if next_word and backward_prob > 0:
                composite_score = (0.55 * forward_prob + 0.45 * backward_prob) * fragment_match
            else:
                composite_score = forward_prob * fragment_match

            scored_candidates.append({
                "text": cand,
                "context_score": round(float(composite_score), 4),
                "model_used": "Bidirectional N-Gram" if (next_word and backward_prob > 0) else f"{self.n}-gram Language Model",
                "forward_prob": round(float(forward_prob), 4),
                "backward_prob": round(float(backward_prob), 4),
            })

        # Sort descending by composite score
        scored_candidates.sort(key=lambda x: x["context_score"], reverse=True)
        top_candidates = scored_candidates[:max_candidates]

        # Apply minimum confidence cutoff
        filtered = [c for c in top_candidates if c["context_score"] >= min_confidence]
        if not filtered and top_candidates:
            filtered = [top_candidates[0]]

        best = filtered[0] if filtered else None
        return {
            "status": "PREDICTED" if best else "LOW_CONFIDENCE",
            "prediction": best["text"] if best else None,
            "confidence_score": best["context_score"] if best else 0.0,
            "candidates": filtered,
            "is_speculative": True,
            "uncertain_span_received": norm_span,
            "disclaimer": (
                "CRITICAL: This text is a context-derived linguistic suggestion estimated from corpus statistics. "
                "Per academic integrity standards, it must NEVER be presented as authentic recovered original manuscript ink."
            ),
            "left_context_used": norm_left,
            "right_context_used": norm_right,
        }

    def save(self, filepath: Path) -> None:
        """Saves fitted predictor model to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n": self.n,
            "vocabulary": dict(self.vocabulary),
            "word_ngrams": {
                "|".join(k): dict(v) for k, v in self.word_ngrams.items()
            },
            "total_words": self.total_words,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: Path) -> None:
        """Loads fitted predictor model from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.n = data["n"]
        self.vocabulary = Counter(data["vocabulary"])
        self.word_ngrams = {
            tuple(k.split("|")): Counter(v) for k, v in data["word_ngrams"].items()
        }
        self.total_words = data["total_words"]
        self.is_fitted = data["is_fitted"]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculates Character Error Rate (CER) via Levenshtein edit distance:
    CER = (Substitutions + Deletions + Insertions) / Total Reference Characters
    """
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    edit_dist = dp[len(ref)][len(hyp)]
    return float(edit_dist / len(ref))


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculates Word Error Rate (WER) via Levenshtein distance on words.
    """
    ref = [w for w in re.split(r"\s+", reference.strip()) if w]
    hyp = [w for w in re.split(r"\s+", hypothesis.strip()) if w]
    if not ref:
        return 0.0 if not hyp else 1.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return float(dp[len(ref)][len(hyp)] / len(ref))
