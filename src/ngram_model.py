"""
N-Gram Language Model with Maximum Likelihood Estimation and Smoothing Techniques.
Milestone 2: Implement n-gram counting and maximum-likelihood probabilities with smoothing.
"""

import math
import json
import pickle
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict, Counter

from src.preprocessor import START_TOKEN, END_TOKEN, UNK_TOKEN


class NGramModel:
    """
    N-gram Language Model supporting arbitrary order n, Maximum Likelihood Estimation,
    and multiple smoothing algorithms:
      1. 'mle' (Raw Maximum Likelihood Estimation)
      2. 'laplace' / 'add_k' (Lidstone add-k / Laplace add-1 smoothing)
      3. 'interpolation' (Jelinek-Mercer linear interpolation across all orders)
      4. 'stupid_backoff' (Brants et al. backoff with penalty factor alpha)
    """

    def __init__(self, n: int = 3, smoothing: str = "interpolation", k: float = 1.0, alpha: float = 0.4):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.smoothing = smoothing.lower()
        self.k = k  # Add-k smoothing parameter
        self.alpha = alpha  # Stupid backoff decay factor

        # Multi-order count tables: order -> context_tuple -> Counter of next_tokens
        # e.g., self.counts[3][("heat", "olive")]["oil"] = 12
        self.counts: Dict[int, Dict[Tuple[str, ...], Counter]] = {
            order: defaultdict(Counter) for order in range(1, n + 1)
        }
        # Context totals: order -> context_tuple -> sum of counts
        self.context_totals: Dict[int, Dict[Tuple[str, ...], int]] = {
            order: defaultdict(int) for order in range(1, n + 1)
        }

        self.vocab: set = set()
        self.total_words: int = 0
        self.lambdas: List[float] = self._default_lambdas(n)

    def _default_lambdas(self, n: int) -> List[float]:
        """Calculates default interpolation weights favoring higher-order n-grams."""
        # e.g. for n=3: weights proportional to order: 1, 2, 4 -> normalized
        weights = [2 ** i for i in range(n)]
        total = sum(weights)
        return [w / total for w in weights]

    def set_interpolation_weights(self, lambdas: List[float]) -> None:
        """Sets custom linear interpolation weights. Must sum to 1.0."""
        if len(lambdas) != self.n:
            raise ValueError(f"Expected {self.n} lambda weights, got {len(lambdas)}")
        s = sum(lambdas)
        if not math.isclose(s, 1.0, rel_tol=1e-4):
            # Normalize
            lambdas = [w / s for w in lambdas]
        self.lambdas = lambdas

    def fit(self, tokenized_sentences: List[List[str]], vocab: Optional[set] = None) -> None:
        """
        Extracts n-grams for all orders from 1 to n and compiles frequency distributions.
        """
        if vocab:
            self.vocab = set(vocab)
        else:
            self.vocab = set()

        for sentence in tokenized_sentences:
            for word in sentence:
                self.vocab.add(word)
                if word not in (START_TOKEN, END_TOKEN):
                    self.total_words += 1

            # Count n-grams for every order from 1 up to self.n
            for order in range(1, self.n + 1):
                for i in range(len(sentence) - order + 1):
                    ngram = sentence[i : i + order]
                    context = tuple(ngram[:-1])
                    target = ngram[-1]

                    self.counts[order][context][target] += 1
                    self.context_totals[order][context] += 1

    def mle_prob(self, word: str, context: Tuple[str, ...], order: Optional[int] = None) -> float:
        """
        Computes unsmoothed Maximum Likelihood Estimation probability:
        P_MLE(w | context) = Count(context, w) / Count(context)
        """
        if order is None:
            order = len(context) + 1

        context_count = self.context_totals[order].get(context, 0)
        if context_count == 0:
            return 0.0

        word_count = self.counts[order][context].get(word, 0)
        return word_count / context_count

    def add_k_prob(self, word: str, context: Tuple[str, ...], k: Optional[float] = None) -> float:
        """
        Computes Add-k (Lidstone / Laplace) smoothed probability:
        P_add_k(w | context) = (Count(context, w) + k) / (Count(context) + k * |V|)
        """
        if k is None:
            k = self.k
        order = self.n
        context_count = self.context_totals[order].get(context, 0)
        word_count = self.counts[order][context].get(word, 0)
        v_size = len(self.vocab) or 1

        return (word_count + k) / (context_count + k * v_size)

    def interpolated_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Computes Jelinek-Mercer linear interpolated probability across all orders:
        P(w | context) = sum_{i=1}^n lambda_i * P_MLE(w | context_{i-1})
        """
        prob = 0.0
        # context contains up to (n - 1) words
        for order in range(1, self.n + 1):
            weight = self.lambdas[order - 1]
            sub_context_len = order - 1
            if sub_context_len == 0:
                sub_context = ()
            else:
                sub_context = context[-sub_context_len:] if len(context) >= sub_context_len else context

            p_mle = self.mle_prob(word, sub_context, order=order)
            prob += weight * p_mle
        return prob

    def stupid_backoff_score(self, word: str, context: Tuple[str, ...], order: Optional[int] = None) -> float:
        """
        Computes Stupid Backoff score (Brants et al., 2007):
        If count(context, word) > 0: score = count(context, word) / count(context)
        Else: score = alpha * stupid_backoff(word, context[1:])
        Base case: unigram count(word) / total_words
        """
        if order is None:
            order = len(context) + 1

        if order == 1:
            # Unigram base case
            unigram_count = self.counts[1][()].get(word, 0)
            return unigram_count / max(self.total_words, 1)

        context_count = self.context_totals[order].get(context, 0)
        word_count = self.counts[order][context].get(word, 0)

        if word_count > 0 and context_count > 0:
            return word_count / context_count
        else:
            shorter_context = context[1:] if len(context) > 1 else ()
            return self.alpha * self.stupid_backoff_score(word, shorter_context, order=order - 1)

    def probability(self, word: str, context: Tuple[str, ...], method: Optional[str] = None) -> float:
        """
        Computes the probability (or score) of `word` given `context` according to the specified method.
        Supported methods: 'mle', 'laplace', 'add_k', 'interpolation', 'stupid_backoff'.
        """
        method = (method or self.smoothing).lower()

        # Truncate context to max (n - 1) tokens
        if len(context) > self.n - 1:
            context = context[-(self.n - 1):]
        # Pad with <s> if shorter
        elif len(context) < self.n - 1:
            context = (START_TOKEN,) * (self.n - 1 - len(context)) + context

        if method == "mle":
            return self.mle_prob(word, context, order=self.n)
        elif method in ("laplace", "add_k"):
            k_val = 1.0 if method == "laplace" else self.k
            return self.add_k_prob(word, context, k=k_val)
        elif method == "interpolation":
            return self.interpolated_prob(word, context)
        elif method == "stupid_backoff":
            return self.stupid_backoff_score(word, context, order=self.n)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def get_distribution(self, context: Tuple[str, ...], method: Optional[str] = None) -> Dict[str, float]:
        """
        Calculates probability or score distribution over all vocabulary tokens given context.
        """
        method = (method or self.smoothing).lower()
        dist: Dict[str, float] = {}

        # Efficient optimization: for MLE and Backoff, check words that occurred with this context or sub-contexts
        for word in self.vocab:
            p = self.probability(word, context, method=method)
            if p > 0:
                dist[word] = p

        # Normalize if it's backoff (so it forms a valid probability distribution)
        if method == "stupid_backoff" and dist:
            total = sum(dist.values())
            if total > 0:
                dist = {w: p / total for w, p in dist.items()}

        return dist

    def calculate_perplexity(self, tokenized_sentences: List[List[str]], method: Optional[str] = None) -> float:
        """
        Calculates test set perplexity:
        Perplexity = exp(- (1/M) * sum_{i=1}^M log P(w_i | context_i))
        Lower perplexity indicates better predictive quality.
        """
        method = method or ("interpolation" if self.smoothing == "mle" else self.smoothing)
        total_log_prob = 0.0
        word_count = 0

        for sentence in tokenized_sentences:
            for i in range(self.n - 1, len(sentence)):
                context = tuple(sentence[i - (self.n - 1) : i])
                target = sentence[i]

                # Don't evaluate on start tokens
                if target == START_TOKEN:
                    continue

                prob = self.probability(target, context, method=method)
                # Safeguard against zero probability for perplexity
                if prob <= 0:
                    prob = 1e-10

                total_log_prob += math.log(prob)
                word_count += 1

        if word_count == 0:
            return float("inf")

        cross_entropy = -total_log_prob / word_count
        return math.exp(cross_entropy)

    def save(self, filepath: str) -> None:
        """Serializes and saves the trained n-gram model to disk."""
        data = {
            "n": self.n,
            "smoothing": self.smoothing,
            "k": self.k,
            "alpha": self.alpha,
            "lambdas": self.lambdas,
            "vocab": list(self.vocab),
            "total_words": self.total_words,
            # Convert counts to serializable dicts with valid JSON string keys
            "counts": {
                order: {
                    json.dumps(list(ctx)): dict(counter)
                    for ctx, counter in ctx_dict.items()
                }
                for order, ctx_dict in self.counts.items()
            },
            "context_totals": {
                order: {
                    json.dumps(list(ctx)): total
                    for ctx, total in ctx_dict.items()
                }
                for order, ctx_dict in self.context_totals.items()
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "NGramModel":
        """Loads and deserializes an n-gram model from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(
            n=data["n"],
            smoothing=data["smoothing"],
            k=data["k"],
            alpha=data["alpha"]
        )
        model.lambdas = data["lambdas"]
        model.vocab = set(data["vocab"])
        model.total_words = data["total_words"]

        for order_str, ctx_dict in data["counts"].items():
            order = int(order_str)
            for ctx_str, counter_dict in ctx_dict.items():
                ctx = tuple(json.loads(ctx_str))
                model.counts[order][ctx] = Counter(counter_dict)

        for order_str, ctx_dict in data["context_totals"].items():
            order = int(order_str)
            for ctx_str, total in ctx_dict.items():
                ctx = tuple(json.loads(ctx_str))
                model.context_totals[order][ctx] = total

        return model
