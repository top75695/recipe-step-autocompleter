"""
Unit tests for RecipeStepGenerator (next-word prediction and autocompletion).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import RecipeTokenizer
from src.ngram_model import NGramModel
from src.generator import RecipeStepGenerator


class TestRecipeStepGenerator(unittest.TestCase):

    def setUp(self):
        self.tokenizer = RecipeTokenizer()
        self.sentences = [
            "preheat the oven to 350 degrees f .",
            "preheat the oven to 400 degrees f .",
            "bake in the preheated oven for 25 minutes .",
            "heat olive oil in a large skillet .",
        ]
        self.tokenized = [
            self.tokenizer.preprocess_sentence(s, n=3) for s in self.sentences
        ]
        self.tokenizer.build_vocab([self.tokenizer.tokenize_raw(s) for s in self.sentences])
        self.model = NGramModel(n=3, smoothing="interpolation")
        self.model.fit(self.tokenized, vocab=self.tokenizer.vocab)
        self.generator = RecipeStepGenerator(self.model, self.tokenizer)

    def test_predict_next_word(self):
        preds = self.generator.predict_next_words("preheat the", top_k=3)
        self.assertTrue(len(preds) > 0)
        top_word, top_score = preds[0]
        self.assertEqual(top_word, "oven")
        self.assertGreater(top_score, 0.5)

    def test_greedy_generation(self):
        result = self.generator.generate_continuation_greedy("preheat the oven", max_tokens=10)
        self.assertIn("to", result.lower())
        self.assertIn("degrees", result.lower())

    def test_suggest_completions(self):
        suggestions = self.generator.suggest_completions("heat olive oil in a", num_suggestions=2)
        self.assertTrue(len(suggestions) >= 1)
        self.assertIn("skillet", suggestions[0].lower())


if __name__ == "__main__":
    unittest.main()
