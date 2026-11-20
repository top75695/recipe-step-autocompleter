"""
Unit tests for NGramModel, MLE, and Smoothing.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import RecipeTokenizer, START_TOKEN, END_TOKEN
from src.ngram_model import NGramModel


class TestNGramModel(unittest.TestCase):

    def setUp(self):
        self.tokenizer = RecipeTokenizer()
        self.sentences = [
            "heat olive oil in a skillet .",
            "heat vegetable oil in a wok .",
            "heat olive oil over medium heat .",
        ]
        self.tokenized = [
            self.tokenizer.preprocess_sentence(s, n=3) for s in self.sentences
        ]
        self.tokenizer.build_vocab([self.tokenizer.tokenize_raw(s) for s in self.sentences])
        self.model = NGramModel(n=3, smoothing="interpolation")
        self.model.fit(self.tokenized, vocab=self.tokenizer.vocab)

    def test_ngram_counts(self):
        # "heat olive" followed by "oil" should count twice
        count = self.model.counts[3][("heat", "olive")]["oil"]
        self.assertEqual(count, 2)

    def test_mle_probability(self):
        # Context ("heat", "olive"): total count is 2, "oil" is 2 -> prob = 1.0
        prob = self.model.mle_prob("oil", ("heat", "olive"))
        self.assertEqual(prob, 1.0)

    def test_smoothing_unseen_word(self):
        # Unseen word should have 0 MLE prob but > 0 for Laplace / Add-k and Interpolation
        mle = self.model.probability("chocolate", ("heat", "olive"), method="mle")
        laplace = self.model.probability("chocolate", ("heat", "olive"), method="laplace")
        self.assertEqual(mle, 0.0)
        self.assertGreater(laplace, 0.0)

    def test_stupid_backoff(self):
        score = self.model.probability("oil", ("heat", "olive"), method="stupid_backoff")
        self.assertGreater(score, 0.0)

    def test_serialization(self):
        temp_path = os.path.join(os.path.dirname(__file__), "temp_model.json")
        self.model.save(temp_path)
        loaded = NGramModel.load(temp_path)
        self.assertEqual(loaded.n, self.model.n)
        self.assertEqual(loaded.counts[3][("heat", "olive")]["oil"], 2)
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
