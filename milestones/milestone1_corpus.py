"""
Milestone 1 Demonstration: Compile a corpus of cooking recipe instructions.
Tokenize and preprocess your text corpus for n-gram analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import RecipeTokenizer, START_TOKEN, END_TOKEN, UNK_TOKEN
from src.corpus import RecipeCorpus


def run_milestone1():
    print("=" * 70)
    print("MILESTONE 1: Corpus Compilation & Preprocessing")
    print("=" * 70)

    # 1. Initialize Tokenizer & Corpus
    tokenizer = RecipeTokenizer(lowercase=True, min_freq=1)
    corpus = RecipeCorpus(tokenizer=tokenizer)

    # 2. Load Corpus
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "recipes_corpus.txt")
    if os.path.exists(data_file):
        count = corpus.load_from_text_file(data_file)
        print(f"✓ Loaded {count} recipe instructions from '{data_file}'.")
    else:
        corpus.load_seed_data()
        print(f"✓ Loaded default seed instructions ({len(corpus.raw_sentences)} steps).")

    # 3. Tokenize and Preprocess
    sample_raw = "Preheat the oven to 350 degrees F (175 degrees C) and grease a 9x13 inch pan."
    raw_tokens = tokenizer.tokenize_raw(sample_raw)
    padded_tokens = tokenizer.preprocess_sentence(sample_raw, n=3)

    print("\n[Sample Preprocessing]")
    print(f"  Raw Text:       '{sample_raw}'")
    print(f"  Tokenized:      {raw_tokens}")
    print(f"  Padded (n=3):   {padded_tokens}")

    # 4. Compile Entire Corpus
    corpus.compile(n=3)
    stats = corpus.get_stats()

    print("\n[Corpus Statistics]")
    print(f"  Total Recipe Steps:      {stats['total_recipe_steps']}")
    print(f"  Total Tokens (padded):   {stats['total_tokens']}")
    print(f"  Vocabulary Size:         {stats['vocab_size']}")
    print(f"  Avg Tokens per Step:     {stats['avg_tokens_per_step']}")

    print("\n[Sample Vocabulary Terms]")
    sample_vocab = [w for w in list(tokenizer.vocab) if w not in (START_TOKEN, END_TOKEN, UNK_TOKEN)][:15]
    print(f"  {sample_vocab}")

    # 5. Detokenization check
    detokenized = tokenizer.detokenize(padded_tokens)
    print(f"\n[Detokenization Reconstruction]")
    print(f"  Reconstructed: '{detokenized}'")

    print("\n✓ Milestone 1 completed successfully!")


if __name__ == "__main__":
    run_milestone1()
