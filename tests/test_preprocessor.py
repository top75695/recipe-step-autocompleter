"""
Unit tests for RecipeTokenizer and Preprocessing.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import RecipeTokenizer, START_TOKEN, END_TOKEN, UNK_TOKEN


class TestRecipeTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = RecipeTokenizer(lowercase=True)

    def test_tokenize_raw_words_and_punctuation(self):
        text = "Preheat oven to 350 degrees F, and grease pan."
        tokens = self.tokenizer.tokenize_raw(text)
        self.assertIn("preheat", tokens)
        self.assertIn("oven", tokens)
        self.assertIn("350", tokens)
        self.assertIn("degrees", tokens)
        self.assertIn("f", tokens)
        self.assertIn(",", tokens)
        self.assertIn(".", tokens)

    def test_fractions_and_temperatures(self):
        text = "Add 1/2 cup of olive oil and heat to 375°F."
        tokens = self.tokenizer.tokenize_raw(text)
        self.assertIn("1/2", tokens)
        self.assertIn("375", tokens)
        self.assertIn("degrees", tokens)

    def test_sentence_splitting(self):
        paragraph = "1. Chop onions. 2. Heat oil in skillet! 3. Cook for 5 minutes."
        sentences = self.tokenizer.split_into_sentences(paragraph)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Chop onions.")

    def test_padding(self):
        tokens = ["heat", "oil"]
        padded = self.tokenizer.pad_sentence(tokens, n=3)
        self.assertEqual(padded, [START_TOKEN, START_TOKEN, "heat", "oil", END_TOKEN])

    def test_detokenize(self):
        tokens = [START_TOKEN, "heat", "olive", "oil", "in", "a", "skillet", ".", END_TOKEN]
        detok = self.tokenizer.detokenize(tokens)
        self.assertEqual(detok, "Heat olive oil in a skillet.")


if __name__ == "__main__":
    unittest.main()
