"""
Recipe Step Continuation and Next-Word Prediction Generator.
Milestone 3: Create a function to generate recipe step continuations.
Develop a function for next-word prediction and text generation.
"""

import math
import random
from typing import List, Tuple, Dict, Any, Optional

from src.preprocessor import RecipeTokenizer, START_TOKEN, END_TOKEN, UNK_TOKEN
from src.ngram_model import NGramModel


class RecipeStepGenerator:
    """
    Handles next-word prediction, beam search, greedy generation,
    and stochastic temperature sampling for recipe instruction autocompletion.
    """

    def __init__(self, model: NGramModel, tokenizer: RecipeTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _extract_context(self, tokens: List[str]) -> Tuple[str, ...]:
        """Extracts the last (n-1) tokens as context for the model."""
        context_len = self.model.n - 1
        if len(tokens) >= context_len:
            return tuple(tokens[-context_len:])
        # Pad with <s> if tokens are fewer than context_len
        padding = [START_TOKEN] * (context_len - len(tokens))
        return tuple(padding + tokens)

    def predict_next_words(
        self,
        partial_text: str,
        top_k: int = 5,
        method: Optional[str] = None,
        exclude_special: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Predicts top-k most likely next words given a partial recipe instruction.
        Returns a list of (word, probability) tuples sorted descending by probability.
        """
        tokens = self.tokenizer.tokenize_raw(partial_text)
        context = self._extract_context(tokens)
        dist = self.model.get_distribution(context, method=method)

        if exclude_special:
            # We don't want to suggest <s> or <unk> directly to users
            dist = {
                word: p
                for word, p in dist.items()
                if word not in (START_TOKEN, UNK_TOKEN)
            }

        sorted_words = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        return sorted_words[:top_k]

    def generate_continuation_greedy(
        self,
        partial_text: str,
        max_tokens: int = 15,
        method: Optional[str] = None
    ) -> str:
        """
        Greedy decoding: iteratively selects the single most probable next token.
        Stops when END_TOKEN or sentence punctuation is reached.
        """
        tokens = self.tokenizer.tokenize_raw(partial_text)
        current_tokens = list(tokens)

        for _ in range(max_tokens):
            context = self._extract_context(current_tokens)
            dist = self.model.get_distribution(context, method=method)

            # Filter special tokens
            valid_candidates = {
                w: p for w, p in dist.items() if w not in (START_TOKEN, UNK_TOKEN)
            }
            if not valid_candidates:
                break

            best_word = max(valid_candidates, key=valid_candidates.get)
            if best_word == END_TOKEN:
                break

            current_tokens.append(best_word)
            if best_word in {".", "!", "?"}:
                break

        return self.tokenizer.detokenize(current_tokens)

    def generate_continuation_sampling(
        self,
        partial_text: str,
        max_tokens: int = 15,
        temperature: float = 0.8,
        top_k: int = 5,
        method: Optional[str] = None
    ) -> str:
        """
        Temperature-scaled top-k sampling for diverse and natural text generation.
        """
        tokens = self.tokenizer.tokenize_raw(partial_text)
        current_tokens = list(tokens)

        for _ in range(max_tokens):
            context = self._extract_context(current_tokens)
            dist = self.model.get_distribution(context, method=method)

            valid_candidates = {
                w: p for w, p in dist.items() if w not in (START_TOKEN, UNK_TOKEN)
            }
            if not valid_candidates:
                break

            # Top-k filtering
            sorted_candidates = sorted(valid_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
            words = [item[0] for item in sorted_candidates]
            probs = [item[1] for item in sorted_candidates]

            # Apply temperature scaling
            if temperature > 0:
                scaled_logits = [math.log(p + 1e-12) / temperature for p in probs]
                max_logit = max(scaled_logits)
                exp_logits = [math.exp(l - max_logit) for l in scaled_logits]
                total_exp = sum(exp_logits)
                weights = [el / total_exp for el in exp_logits]
            else:
                weights = [1.0 if i == 0 else 0.0 for i in range(len(words))]

            chosen_word = random.choices(words, weights=weights, k=1)[0]
            if chosen_word == END_TOKEN:
                break

            current_tokens.append(chosen_word)
            if chosen_word in {".", "!", "?"}:
                break

        return self.tokenizer.detokenize(current_tokens)

    def generate_continuation_beam(
        self,
        partial_text: str,
        beam_width: int = 3,
        max_tokens: int = 12,
        method: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Beam search decoding: tracks the top-k most probable sequence hypotheses.
        Returns a list of (completed_sentence, log_probability) ranked descending.
        """
        tokens = self.tokenizer.tokenize_raw(partial_text)
        # Each beam item: (tokens_list, accumulated_log_prob, is_done)
        beams: List[Tuple[List[str], float, bool]] = [(tokens, 0.0, False)]

        for _ in range(max_tokens):
            all_candidates: List[Tuple[List[str], float, bool]] = []
            all_done = True

            for seq, score, done in beams:
                if done:
                    all_candidates.append((seq, score, True))
                    continue

                all_done = False
                context = self._extract_context(seq)
                dist = self.model.get_distribution(context, method=method)
                valid = {
                    w: p for w, p in dist.items() if w not in (START_TOKEN, UNK_TOKEN)
                }

                if not valid:
                    all_candidates.append((seq, score, True))
                    continue

                # Top candidates for this beam
                top_items = sorted(valid.items(), key=lambda x: x[1], reverse=True)[:beam_width]
                for word, prob in top_items:
                    new_score = score + math.log(prob + 1e-12)
                    is_finished = (word == END_TOKEN or word in {".", "!", "?"})
                    new_seq = seq if word == END_TOKEN else seq + [word]
                    all_candidates.append((new_seq, new_score, is_finished))

            if all_done:
                break

            # Keep top beam_width hypotheses (normalized by length to prevent bias against longer sentences)
            all_candidates.sort(key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)
            beams = all_candidates[:beam_width]

        results = []
        for seq, score, _ in beams:
            sentence = self.tokenizer.detokenize(seq)
            results.append((sentence, round(score, 3)))
        return results

    def suggest_completions(
        self,
        partial_text: str,
        num_suggestions: int = 3,
        max_tokens: int = 12,
        method: Optional[str] = None
    ) -> List[str]:
        """
        Generates distinct recipe step autocompletion suggestions for the user.
        Combines greedy and beam search to ensure high relevance and variety.
        """
        beam_results = self.generate_continuation_beam(
            partial_text,
            beam_width=max(num_suggestions, 3),
            max_tokens=max_tokens,
            method=method
        )
        suggestions: List[str] = []
        seen = set()

        for sentence, _ in beam_results:
            if sentence and sentence not in seen:
                suggestions.append(sentence)
                seen.add(sentence)

        # Fallback or fill with greedy / sampling if beam didn't produce enough distinct outputs
        if len(suggestions) < num_suggestions:
            greedy_sent = self.generate_continuation_greedy(partial_text, max_tokens=max_tokens, method=method)
            if greedy_sent and greedy_sent not in seen:
                suggestions.append(greedy_sent)
                seen.add(greedy_sent)

        # If still need more, use sampling
        attempts = 0
        while len(suggestions) < num_suggestions and attempts < 10:
            attempts += 1
            sample_sent = self.generate_continuation_sampling(
                partial_text, max_tokens=max_tokens, temperature=0.7, method=method
            )
            if sample_sent and sample_sent not in seen:
                suggestions.append(sample_sent)
                seen.add(sample_sent)

        return suggestions[:num_suggestions]
