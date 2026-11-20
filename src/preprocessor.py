"""
Preprocessor & Tokenizer for Recipe Step Text Corpus.
Milestone 1: Tokenize and preprocess your text corpus for n-gram analysis.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter


START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


class RecipeTokenizer:
    """
    Tokenizer tailored for cooking recipe instructions.
    Handles culinary terms, measurements, fractions, temperatures, and sentence boundaries.
    """

    # Regex pattern to capture words, contractions, fractions (e.g., 1/2), numbers, and punctuation
    TOKEN_REGEX = re.compile(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+/\d+|\d+(?:\.\d+)?|[.,!?;:]"
    )

    def __init__(self, lowercase: bool = True, handle_unknown: bool = True, min_freq: int = 1):
        self.lowercase = lowercase
        self.handle_unknown = handle_unknown
        self.min_freq = min_freq
        self.vocab: Set[str] = set()
        self.vocab_counts: Counter = Counter()
        self.special_tokens = {START_TOKEN, END_TOKEN, UNK_TOKEN}

    def tokenize_raw(self, text: str) -> List[str]:
        """
        Tokenizes raw text into individual word and punctuation tokens.
        """
        if self.lowercase:
            text = text.lower()
        
        # Normalize various unicode apostrophes / quotes
        text = text.replace("’", "'").replace("`", "'")
        # Normalize degree symbols: e.g. 350°F -> 350 degrees f
        text = re.sub(r'(\d+)\s*°\s*([fcFC])', r'\1 degrees \2', text)
        text = re.sub(r'(\d+)\s*°', r'\1 degrees', text)

        tokens = self.TOKEN_REGEX.findall(text)
        return tokens

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits a recipe paragraph or multi-step instruction into discrete step sentences.
        """
        # Split on sentence boundaries: period, exclamation, question mark, newline, or step numbering (e.g., "1.", "Step 2:")
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+|(?:^|\s+)\d+[\.)]\s*', text)
        cleaned = []
        for s in raw_sentences:
            s = s.strip()
            # Remove any leading step numbers like "1 - ", "Step 1:"
            s = re.sub(r'^(?:step\s*\d+[:\-.]?|\d+[:\-.]?)\s*', '', s, flags=re.IGNORECASE).strip()
            if len(s) > 2:  # ignore empty/trivial snippets
                cleaned.append(s)
        return cleaned

    def build_vocab(self, corpus_sentences: List[List[str]]) -> None:
        """
        Builds vocabulary from tokenized sentences, tracking word frequencies.
        Applies min_freq cutoff if configured.
        """
        self.vocab_counts = Counter()
        for sent in corpus_sentences:
            for token in sent:
                if token not in self.special_tokens:
                    self.vocab_counts[token] += 1

        self.vocab = set(self.special_tokens)
        for token, count in self.vocab_counts.items():
            if count >= self.min_freq:
                self.vocab.add(token)

    def pad_sentence(self, tokens: List[str], n: int = 3) -> List[str]:
        """
        Pads a tokenized sentence with start and end tokens for an n-gram model of order n.
        For order n, we prepend (n-1) START_TOKEN and append 1 END_TOKEN.
        Example for n=3: ['heat', 'oil'] -> ['<s>', '<s>', 'heat', 'oil', '</s>']
        """
        start_padding = [START_TOKEN] * (n - 1)
        end_padding = [END_TOKEN]
        return start_padding + tokens + end_padding

    def preprocess_sentence(self, sentence: str, n: int = 3, replace_unk: bool = False) -> List[str]:
        """
        Preprocesses and tokenizes a single sentence, then pads it with start and end tokens.
        """
        tokens = self.tokenize_raw(sentence)
        if replace_unk and self.vocab:
            tokens = [t if t in self.vocab else UNK_TOKEN for t in tokens]
        return self.pad_sentence(tokens, n=n)

    def detokenize(self, tokens: List[str]) -> str:
        """
        Converts a list of tokens back into human-readable text.
        Removes special symbols like <s> and </s>, fixes spaces before punctuation,
        and capitalizes the first character.
        """
        filtered = [t for t in tokens if t not in (START_TOKEN, END_TOKEN, UNK_TOKEN)]
        if not filtered:
            return ""

        result = []
        for i, token in enumerate(filtered):
            if token in {".", ",", "!", "?", ";", ":"}:
                if result:
                    result[-1] = result[-1] + token
                else:
                    result.append(token)
            elif token.startswith("'") and result:
                result[-1] = result[-1] + token
            else:
                result.append(token)

        sentence = " ".join(result)
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        return sentence
