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
        self.reverse_word_ngrams: Dict[str, Counter] = defaultdict(Counter)
        self.char_ngrams: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocabulary: Counter = Counter()
        self.total_words: int = 0
        self.is_fitted: bool = False

    @staticmethod
    def is_placeholder(span: Optional[str]) -> bool:
        """Determines if a span represents an explicit blank/missing placeholder."""
        if not span:
            return True
        s = span.strip()
        # Common placeholder patterns: [MISSING], [DAMAGED], [?], ___, ???, etc.
        pattern = r"^(\[?(MISSING|DAMAGED|UNCERTAIN|UNKNOWN|BLANK|\?)\]?|[\_\-\*\?\.]{1,10})$"
        return bool(re.match(pattern, s, re.IGNORECASE))

    @staticmethod
    def detect_script(text: str) -> str:
        """Detects whether text primarily contains Devanagari or Latin script."""
        if re.search(r"[\u0900-\u097F]", text):
            return "devanagari"
        if re.search(r"[A-Za-z]", text):
            return "latin"
        return "all"

    def train_on_corpus(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Trains n-gram statistical language model on text sentences (Devanagari, Sanskrit, English).
        Builds both forward n-grams and reverse n-grams for bidirectional context scoring.
        """
        word_count = 0
        for sentence in sentences:
            # Tokenize words preserving Devanagari and Latin characters
            tokens = [w for w in re.split(r"[\s,।॥;:]+", sentence) if w]
            if not tokens:
                continue
            word_count += len(tokens)
            for w in tokens:
                self.vocabulary[w] += 1

            # Word forward n-grams
            padded_words = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for i in range(len(padded_words) - self.n + 1):
                ctx = tuple(padded_words[i : i + self.n - 1])
                target = padded_words[i + self.n - 1]
                self.word_ngrams[ctx][target] += 1

            # Bigram forward & backward for bidirectional transitions
            for i in range(len(tokens) - 1):
                w_curr, w_next = tokens[i], tokens[i + 1]
                self.word_ngrams[(w_curr,)][w_next] += 1
                self.word_ngrams[(w_curr.lower(),)][w_next] += 1
                self.reverse_word_ngrams[w_next][w_curr] += 1
                self.reverse_word_ngrams[w_next.lower()][w_curr] += 1

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

    @classmethod
    def parse_sentence_context(
        cls,
        sentence: str,
        uncertain_span: Optional[str] = None,
        context_window_words: int = 4,
    ) -> Tuple[str, str, str, str]:
        """
        Parses a complete sentence to identify the uncertain span, left context, and right context.
        Supports placeholders like [MISSING], [DAMAGED], ___, or explicit target words.
        Returns: (left_context, actual_span, right_context, original_sentence)
        """
        import unicodedata
        s = unicodedata.normalize("NFC", (sentence or "").strip())
        if not s:
            return "", uncertain_span or "", "", ""

        target = (uncertain_span or "").strip()

        # Case 1: Target span is specified and exists in the sentence
        if target and target in s:
            parts = s.split(target, 1)
            left_part, right_part = parts[0].strip(), parts[1].strip()
            left_w = [w for w in re.split(r"[\s,।॥]+", left_part) if w][-context_window_words:]
            right_w = [w for w in re.split(r"[\s,।॥]+", right_part) if w][:context_window_words]
            return " ".join(left_w), target, " ".join(right_w), s

        # Case 2: Sentence contains an explicit placeholder like [MISSING], [DAMAGED], [?], ___
        placeholder_match = re.search(
            r"(\[?(MISSING|DAMAGED|UNCERTAIN|UNKNOWN|BLANK|\?)\]?|[\_\-\*\?]{2,})",
            s,
            re.IGNORECASE,
        )
        if placeholder_match:
            found_span = placeholder_match.group(0)
            parts = s.split(found_span, 1)
            left_part, right_part = parts[0].strip(), parts[1].strip()
            left_w = [w for w in re.split(r"[\s,।॥]+", left_part) if w][-context_window_words:]
            right_w = [w for w in re.split(r"[\s,।॥]+", right_part) if w][:context_window_words]
            return " ".join(left_w), found_span, " ".join(right_w), s

        # Case 3: Fallback token split
        words = [w for w in re.split(r"(\s+|[।॥,;]+)", s) if w.strip()]
        if target:
            for i, w in enumerate(words):
                if w == target or target in w:
                    left_slice = words[max(0, i - context_window_words) : i]
                    right_slice = words[i + 1 : min(len(words), i + 1 + context_window_words)]
                    return " ".join(left_slice), w, " ".join(right_slice), s

        return "", target or "", "", s

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

        target_idx = -1
        if span_index is not None and 0 <= span_index < len(words):
            target_idx = span_index
        else:
            for i, w in enumerate(words):
                if w == normalized_target or normalized_target in w or w in normalized_target:
                    target_idx = i
                    break

        if target_idx == -1:
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

    @classmethod
    def reconstruct_sentence(
        cls,
        original_sentence: Optional[str],
        left_context: str,
        replacement: str,
        right_context: str,
        uncertain_span: Optional[str] = None,
    ) -> str:
        """
        Reconstructs the full sentence with the predicted replacement cleanly inserted.
        Preserves original sentence structure, casing, and punctuation where available.
        """
        if original_sentence and uncertain_span and uncertain_span in original_sentence:
            # Direct substitution of the uncertain span in the original sentence
            return original_sentence.replace(uncertain_span, replacement, 1)

        # Fallback: assemble from left_context, replacement, and right_context
        left = left_context.strip()
        right = right_context.strip()
        if left and right:
            # Handle punctuation spacing before appending right context
            if re.match(r"^[।॥,;:!?.']", right):
                return f"{left} {replacement}{right}".strip()
            return f"{left} {replacement} {right}".strip()
        elif left:
            return f"{left} {replacement}".strip()
        elif right:
            return f"{replacement} {right}".strip()
        return replacement

    def predict_missing(
        self,
        left_context: str = "",
        right_context: str = "",
        uncertain_span: Optional[str] = None,
        sentence: Optional[str] = None,
        max_candidates: int = 4,
        min_confidence: float = 0.01,
        allow_multi_word: bool = True,
    ) -> Dict[str, Any]:
        """Convenience method for predict_missing_span supporting optional arguments."""
        return self.predict_missing_span(
            left_context=left_context,
            right_context=right_context,
            uncertain_span=uncertain_span,
            sentence=sentence,
            max_candidates=max_candidates,
            min_confidence=min_confidence,
            allow_multi_word=allow_multi_word,
        )

    def predict_missing_span(
        self,
        left_context: str,
        right_context: str,
        uncertain_span: Optional[str] = None,
        sentence: Optional[str] = None,
        max_candidates: int = 4,
        min_confidence: float = 0.01,
        allow_multi_word: bool = True,
    ) -> Dict[str, Any]:
        """
        Suggests candidate completions for a missing/damaged span between left_context and right_context,
        optionally constrained by surviving character fragments in uncertain_span or sentence context.
        
        Returns structured prediction marked as SPECULATIVE with explicit uncertainty flags.
        """
        import unicodedata
        norm_left = unicodedata.normalize("NFC", (left_context or "").strip())
        norm_right = unicodedata.normalize("NFC", (right_context or "").strip())
        norm_span = unicodedata.normalize("NFC", (uncertain_span or "").strip())
        orig_sentence = unicodedata.normalize("NFC", (sentence or "").strip())

        # If full sentence was provided, parse and reconcile contexts if needed
        if orig_sentence and (not norm_left or not norm_right or not norm_span):
            p_left, p_span, p_right, _ = self.parse_sentence_context(orig_sentence, norm_span)
            norm_left = norm_left or p_left
            norm_span = norm_span or p_span
            norm_right = norm_right or p_right

        if not self.is_fitted:
            return {
                "status": "UNAVAILABLE",
                "reason": "Context predictor model is not yet fitted on textual corpus.",
                "candidates": [],
                "prediction": None,
                "confidence_score": 0.0,
                "original_sentence": orig_sentence,
                "reconstructed_sentence": None,
                "is_speculative": True,
            }

        # Determine placeholder vs suspicious word vs partial fragment
        is_placeholder_span = self.is_placeholder(norm_span)
        has_character_clue = False
        clean_prefix = ""
        if not is_placeholder_span and norm_span:
            # Check if there are partial character fragments like "य[?]स्य" or "vis[?]ed"
            if "?" in norm_span or "*" in norm_span or "[" in norm_span:
                has_character_clue = True
                clean_prefix = re.sub(r"[\[\]\?\.\_\s\(\)\*]+", "", norm_span)
            else:
                # Whole word marked as uncertain (e.g. "visited" in "The king visited the market")
                clean_prefix = norm_span

        # Detect script to avoid cross-language candidate pollution
        detected_script = self.detect_script(f"{norm_left} {norm_right} {norm_span}")

        # 1. Left-context n-gram extraction
        left_words = [w for w in re.split(r"[\s,।॥]+", norm_left) if w]
        if not left_words:
            ctx = ("<s>",) if self.n == 2 else ("<s>", "<s>")
            prev_word = None
        elif len(left_words) == 1:
            ctx = ("<s>", left_words[-1]) if self.n == 3 else (left_words[-1],)
            prev_word = left_words[-1]
        else:
            ctx = tuple(left_words[-(self.n - 1) :])
            prev_word = left_words[-1]

        candidates_counter = Counter(self.word_ngrams.get(ctx, Counter()))
        # Back off to single previous word if trigram is sparse
        if len(candidates_counter) < max_candidates and prev_word:
            backoff_ctx = (prev_word,)
            for w, c in self.word_ngrams.get(backoff_ctx, Counter()).items():
                candidates_counter[w] += c
            # Also try lowercase match for English words
            if prev_word.lower() != prev_word:
                for w, c in self.word_ngrams.get((prev_word.lower(),), Counter()).items():
                    candidates_counter[w] += c

        # 2. Right-context word for bidirectional scoring
        right_words = [w for w in re.split(r"[\s,।॥]+", norm_right) if w]
        next_word = right_words[0] if right_words else None

        # 3. Candidate generation pool
        candidate_pool = set()
        for w, _ in candidates_counter.most_common(30):
            if w not in ["<s>", "</s>"]:
                candidate_pool.add(w)

        # Right-context reverse transitions
        if next_word:
            for w, _ in self.reverse_word_ngrams.get(next_word, Counter()).most_common(20):
                if w not in ["<s>", "</s>"]:
                    candidate_pool.add(w)
            if next_word.lower() != next_word:
                for w, _ in self.reverse_word_ngrams.get(next_word.lower(), Counter()).most_common(20):
                    if w not in ["<s>", "</s>"]:
                        candidate_pool.add(w)

        # Filter candidate pool by detected script to prevent mixing Devanagari and Latin
        if detected_script in ["devanagari", "latin"]:
            candidate_pool = {
                w for w in candidate_pool if self.detect_script(w) in [detected_script, "all"]
            }

        # If pool is still sparse, back off to script-specific vocabulary
        if len(candidate_pool) < max_candidates * 2:
            for w, _ in self.vocabulary.most_common(max_candidates * 10):
                if w not in ["<s>", "</s>"]:
                    if detected_script == "all" or self.detect_script(w) == detected_script:
                        candidate_pool.add(w)

        # Multi-word candidate generation (e.g. for [DAMAGED] spans)
        multi_word_candidates = []
        if allow_multi_word and (norm_span.upper() in ["[DAMAGED]", "DAMAGED"] or not is_placeholder_span):
            for w1 in list(candidate_pool)[:5]:
                next_cands = self.word_ngrams.get((w1,), Counter())
                for w2, count in next_cands.most_common(3):
                    if w2 not in ["<s>", "</s>"]:
                        if detected_script == "all" or self.detect_script(w2) == detected_script:
                            multi_word_candidates.append(f"{w1} {w2}")

        # If partial fragment with character clue exists (e.g. य[?]स्य), boost words matching fragment
        if has_character_clue and clean_prefix and len(clean_prefix) >= 1:
            for w in self.vocabulary:
                if w.startswith(clean_prefix) or clean_prefix in w:
                    if detected_script == "all" or self.detect_script(w) == detected_script:
                        candidate_pool.add(w)

        all_candidates_to_score = list(candidate_pool) + multi_word_candidates
        if not all_candidates_to_score:
            return {
                "status": "INSUFFICIENT_CONTEXT",
                "reason": "Context is too sparse or unseen in the trained language model to provide a statistically reliable prediction.",
                "candidates": [],
                "prediction": None,
                "confidence_score": 0.0,
                "original_sentence": orig_sentence,
                "reconstructed_sentence": None,
                "is_speculative": True,
                "surrounding_context": {
                    "left_context": norm_left,
                    "right_context": norm_right,
                    "uncertain_span": norm_span,
                },
            }

        # 4. Bidirectional scoring for each candidate
        total_ctx = sum(candidates_counter.values()) if candidates_counter else 0
        total_vocab_occ = max(sum(self.vocabulary.values()), 1)
        scored_candidates = []

        for cand in all_candidates_to_score:
            first_w = cand.split()[0]
            last_w = cand.split()[-1]

            # A. Forward probability P(cand | left_context)
            p_left = (candidates_counter.get(first_w, 0) / total_ctx) if total_ctx > 0 else 0.0
            p_unigram = self.vocabulary.get(first_w, 0) / total_vocab_occ
            forward_prob = 0.80 * p_left + 0.20 * p_unigram

            # B. Backward probability P(right_context | cand)
            backward_prob = 0.0
            if next_word:
                next_counter = self.word_ngrams.get((last_w,), Counter())
                if next_counter:
                    sub_count = next_counter.get(next_word, 0) + next_counter.get(next_word.lower(), 0)
                    backward_prob = sub_count / max(sum(next_counter.values()), 1)
                else:
                    rev_counter = self.reverse_word_ngrams.get(next_word, Counter())
                    if rev_counter:
                        sub_count = rev_counter.get(last_w, 0)
                        backward_prob = sub_count / max(sum(rev_counter.values()), 1)

            # C. Fragment similarity bonus
            fragment_match = 1.0
            if has_character_clue and clean_prefix:
                if cand == clean_prefix:
                    fragment_match = 2.0
                elif cand.startswith(clean_prefix):
                    fragment_match = 1.6
                elif clean_prefix in cand:
                    fragment_match = 1.3
                else:
                    fragment_match = 0.4

            # Composite bidirectional score
            if next_word and backward_prob > 0:
                composite_score = (0.50 * forward_prob + 0.50 * backward_prob) * fragment_match
            else:
                composite_score = forward_prob * fragment_match

            # Candidate reconstructed sentence
            cand_reconstructed = self.reconstruct_sentence(
                original_sentence=orig_sentence,
                left_context=norm_left,
                replacement=cand,
                right_context=norm_right,
                uncertain_span=norm_span,
            )

            scored_candidates.append({
                "text": cand,
                "context_score": round(float(composite_score), 4),
                "model_used": "Bidirectional N-Gram" if (next_word and backward_prob > 0) else f"{self.n}-gram Language Model",
                "forward_prob": round(float(forward_prob), 4),
                "backward_prob": round(float(backward_prob), 4),
                "reconstructed_sentence": cand_reconstructed,
            })

        # Sort descending by composite score
        scored_candidates.sort(key=lambda x: x["context_score"], reverse=True)
        top_candidates = scored_candidates[:max_candidates]

        # Calculate normalized relative confidence for top candidate
        top_score_sum = sum(c["context_score"] for c in top_candidates)
        if top_score_sum > 0:
            for c in top_candidates:
                c["relative_confidence"] = round(float(c["context_score"] / top_score_sum), 3)
        else:
            for c in top_candidates:
                c["relative_confidence"] = 0.25

        best = top_candidates[0] if top_candidates else None
        top_confidence = best["relative_confidence"] if best and best["context_score"] >= min_confidence else 0.0

        final_reconstructed = best["reconstructed_sentence"] if best else orig_sentence

        return {
            "status": "PREDICTED" if (best and best["context_score"] >= min_confidence) else "LOW_CONFIDENCE",
            "prediction": best["text"] if best else None,
            "confidence_score": top_confidence,
            "original_sentence": orig_sentence,
            "reconstructed_sentence": final_reconstructed,
            "candidates": top_candidates,
            "is_speculative": True,
            "uncertain_span_received": norm_span,
            "detected_script": detected_script,
            "disclaimer": (
                "CRITICAL: This text is a context-derived linguistic suggestion estimated from corpus statistics. "
                "Per academic integrity standards, it must NEVER be presented as authentic recovered original manuscript ink."
            ),
            "surrounding_context": {
                "left_context": norm_left,
                "right_context": norm_right,
                "uncertain_span": norm_span,
            },
        }

    def score_candidates(
        self,
        candidates: List[str],
        left_context: str = "",
        right_context: str = "",
    ) -> Dict[str, float]:
        """
        Scores a specific list of candidate words based on left and right context n-gram probabilities.
        Returns a dictionary mapping candidate -> normalized score in [0.0, 1.0].
        """
        if not candidates:
            return {}

        left_tokens = [w for w in left_context.strip().split() if w]
        right_tokens = [w for w in right_context.strip().split() if w]
        prev_word = left_tokens[-1] if left_tokens else None
        next_word = right_tokens[0] if right_tokens else None

        candidates_counter = self.word_ngrams.get((prev_word,), Counter()) if prev_word else Counter()
        total_ctx = sum(candidates_counter.values()) if candidates_counter else 0
        total_vocab = max(sum(self.vocabulary.values()), 1)

        scores = {}
        for cand in candidates:
            first_w = cand.split()[0]
            last_w = cand.split()[-1]

            # Forward probability
            p_fwd = (candidates_counter.get(first_w, 0) / total_ctx) if total_ctx > 0 else 0.0
            p_uni = self.vocabulary.get(first_w, 0) / total_vocab
            fwd_score = 0.70 * p_fwd + 0.30 * p_uni

            # Backward probability
            bwd_score = 0.0
            if next_word:
                next_counter = self.word_ngrams.get((last_w,), Counter())
                if next_counter:
                    tot_next = sum(next_counter.values())
                    bwd_score = next_counter.get(next_word, 0) / tot_next if tot_next > 0 else 0.0
                else:
                    rev_counter = self.reverse_word_ngrams.get(next_word, Counter())
                    tot_rev = sum(rev_counter.values())
                    bwd_score = rev_counter.get(last_w, 0) / tot_rev if tot_rev > 0 else 0.0

            combined = (0.60 * fwd_score + 0.40 * bwd_score) if next_word else fwd_score
            scores[cand] = float(combined)

        max_s = max(scores.values()) if scores else 0.0
        if max_s > 0:
            return {k: round(v / max_s, 3) for k, v in scores.items()}
        return {k: 0.1 for k in scores}

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
        """Loads fitted predictor model from JSON and rebuilds reverse n-grams."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure word_ngrams is a defaultdict(Counter) so further training succeeds
        loaded_ngrams = {
            tuple(k.split("|")): Counter(v) for k, v in data["word_ngrams"].items()
        }
        self.word_ngrams = defaultdict(Counter, loaded_ngrams)
        self.total_words = data["total_words"]
        self.is_fitted = data["is_fitted"]

        # Rebuild reverse n-grams
        self.reverse_word_ngrams = defaultdict(Counter)
        for ctx, counter in self.word_ngrams.items():
            if len(ctx) >= 1:
                prev_w = ctx[-1]
                for next_w, count in counter.items():
                    self.reverse_word_ngrams[next_w][prev_w] += count
                    self.reverse_word_ngrams[next_w.lower()][prev_w] += count

    def score_candidates(
        self,
        candidates: List[str],
        left_context: str = "",
        right_context: str = "",
    ) -> Dict[str, float]:
        """
        Calculates linguistic transition likelihood for a set of candidate words
        given surrounding left and right context.
        Returns a mapping from candidate word to a normalized score in [0.0, 1.0].
        """
        if not candidates:
            return {}

        left_words = [w for w in re.split(r"[\s,।॥;:]+", left_context.strip()) if w]
        right_words = [w for w in re.split(r"[\s,।॥;:]+", right_context.strip()) if w]

        prev_word = left_words[-1] if left_words else None
        next_word = right_words[0] if right_words else None

        total_vocab_occ = max(sum(self.vocabulary.values()), 1)
        raw_scores: Dict[str, float] = {}

        for cand in candidates:
            cand_clean = cand.strip()
            if not cand_clean:
                continue

            first_w = cand_clean.split()[0]
            last_w = cand_clean.split()[-1]

            # Unigram prior
            p_unigram = (self.vocabulary.get(first_w, 0) + self.vocabulary.get(first_w.lower(), 0)) / total_vocab_occ

            # Forward transition P(cand | left_context)
            p_forward = 0.0
            if prev_word:
                prev_counter = self.word_ngrams.get((prev_word,), Counter())
                if not prev_counter and prev_word.lower() != prev_word:
                    prev_counter = self.word_ngrams.get((prev_word.lower(),), Counter())
                if prev_counter:
                    count = prev_counter.get(first_w, 0) + prev_counter.get(first_w.lower(), 0)
                    p_forward = count / max(sum(prev_counter.values()), 1)

            # Backward transition P(right_context | cand)
            p_backward = 0.0
            if next_word:
                cand_counter = self.word_ngrams.get((last_w,), Counter())
                if cand_counter:
                    count = cand_counter.get(next_word, 0) + cand_counter.get(next_word.lower(), 0)
                    p_backward = count / max(sum(cand_counter.values()), 1)
                else:
                    rev_counter = self.reverse_word_ngrams.get(next_word, Counter())
                    if rev_counter:
                        count = rev_counter.get(last_w, 0) + rev_counter.get(last_w.lower(), 0)
                        p_backward = count / max(sum(rev_counter.values()), 1)

            # Composite weighting
            score = 0.45 * p_forward + 0.35 * p_backward + 0.20 * p_unigram
            raw_scores[cand] = max(score, 0.01)

        # Normalize relative scores to [0.0, 1.0]
        max_s = max(raw_scores.values()) if raw_scores else 1.0
        if max_s > 0:
            return {c: round(s / max_s, 4) for c, s in raw_scores.items()}
        return {c: 0.5 for c in candidates}


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


# Baseline training corpus for standalone singleton
_DEFAULT_BASE_CORPUS = [
    "The king entered the temple to offer prayers.",
    "The king protected the temple and its treasury.",
    "The king visited the market before returning to the palace.",
    "The king entered the market and greeted the citizens.",
    "The manuscript was preserved within the temple archives.",
    "The noble citizens gathered at the market square.",
    "The sacred verses were recited by the learned scholars.",
    "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
    "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
    "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम् । होतारं रत्नधातमम् ॥",
    "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत । अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥",
    "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज । अहं त्वां सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः ॥",
    "ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥",
]

# Module singleton instance
context_predictor = ContextTextPredictor(n=3)
context_predictor.train_on_corpus(_DEFAULT_BASE_CORPUS)

