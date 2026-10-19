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

    def predict_missing_span(
        self,
        left_context: str,
        right_context: str,
        max_candidates: int = 3,
        min_confidence: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Suggests candidate completions for a missing/damaged span between left_context and right_context.
        
        Returns structured prediction marked as SPECULATIVE with explicit uncertainty flags.
        """
        if not self.is_fitted:
            return {
                "status": "UNAVAILABLE",
                "reason": "Context predictor model is not yet fitted on textual corpus.",
                "candidates": [],
                "prediction": None,
                "is_speculative": True,
            }

        left_words = [w for w in re.split(r"[\s,।॥]+", left_context) if w]
        if not left_words:
            ctx = ("<s>",) if self.n == 2 else ("<s>", "<s>")
        elif len(left_words) == 1:
            ctx = ("<s>", left_words[-1]) if self.n == 3 else (left_words[-1],)
        else:
            ctx = tuple(left_words[-(self.n - 1) :])

        candidates_counter = self.word_ngrams.get(ctx, Counter())

        if not candidates_counter:
            # Fallback to unigram backoff
            top_unigrams = self.vocabulary.most_common(max_candidates)
            if not top_unigrams:
                return {
                    "status": "INSUFFICIENT_CONTEXT",
                    "reason": "Context is too sparse to provide a mathematically defensible prediction.",
                    "candidates": [],
                    "prediction": None,
                    "is_speculative": True,
                }
            candidates = []
            total_vocab_occ = sum(self.vocabulary.values())
            for word, count in top_unigrams:
                if word not in ["<s>", "</s>"]:
                    candidates.append({
                        "text": word,
                        "context_score": round(count / total_vocab_occ, 3),
                        "model_used": "Unigram Backoff",
                    })
        else:
            total_ctx = sum(candidates_counter.values())
            candidates = []
            for word, count in candidates_counter.most_common(max_candidates):
                if word not in ["<s>", "</s>"]:
                    candidates.append({
                        "text": word,
                        "context_score": round(count / total_ctx, 3),
                        "model_used": f"{self.n}-gram Language Model",
                    })

        # Apply minimum confidence cutoff
        filtered = [c for c in candidates if c["context_score"] >= min_confidence]
        if not filtered:
            return {
                "status": "LOW_CONFIDENCE",
                "reason": f"Top candidate confidence below threshold ({min_confidence}). No reliable prediction.",
                "candidates": candidates,
                "prediction": None,
                "is_speculative": True,
            }

        best = filtered[0]
        return {
            "status": "PREDICTED",
            "prediction": best["text"],
            "confidence_score": best["context_score"],
            "candidates": filtered,
            "is_speculative": True,
            "disclaimer": "CRITICAL: This text is a context-derived linguistic suggestion. It must NEVER be treated as verified historical ink without independent manuscript confirmation.",
            "left_context_used": left_context,
            "right_context_used": right_context,
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
