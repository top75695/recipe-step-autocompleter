"""
Milestone 2 Demonstration: Implement n-gram counting and maximum-likelihood probabilities with smoothing.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import RecipeTokenizer
from src.corpus import RecipeCorpus
from src.ngram_model import NGramModel


def run_milestone2():
    print("=" * 70)
    print("MILESTONE 2: N-Gram Counting, MLE & Smoothing Techniques")
    print("=" * 70)

    # 1. Load and compile corpus
    corpus = RecipeCorpus()
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "recipes_corpus.txt")
    if os.path.exists(data_file):
        corpus.load_from_text_file(data_file)
    else:
        corpus.load_seed_data()
    corpus.compile(n=3)

    # 2. Train N-gram Model (Trigram with interpolation)
    print("\n[Training Trigram (n=3) Language Model]")
    model = NGramModel(n=3, smoothing="interpolation")
    model.fit(corpus.tokenized_sentences, vocab=corpus.tokenizer.vocab)

    print(f"✓ Total words processed: {model.total_words}")
    print(f"✓ Contexts tracked for order 1 (unigram): {len(model.counts[1])}")
    print(f"✓ Contexts tracked for order 2 (bigram):  {len(model.counts[2])}")
    print(f"✓ Contexts tracked for order 3 (trigram): {len(model.counts[3])}")

    # 3. Demonstrate N-gram Counting & MLE
    test_context = ("olive", "oil")
    test_target = "in"
    c_context = model.context_totals[3].get(test_context, 0)
    c_target = model.counts[3][test_context].get(test_target, 0)
    mle_p = model.mle_prob(test_target, test_context)

    print(f"\n[Count & MLE Calculation]")
    print(f"  Context:                 {test_context}")
    print(f"  Target word:             '{test_target}'")
    print(f"  Count(context, target):  {c_target}")
    print(f"  Count(context):          {c_context}")
    print(f"  P_MLE('{test_target}' | {test_context}) = {c_target} / {c_context} = {mle_p:.4f}")

    # 4. Compare Smoothing Techniques on seen vs unseen transitions
    print("\n[Comparison of Smoothing Techniques]")
    scenarios = [
        (("olive", "oil"), "in", "Frequent transition"),
        (("preheat", "the"), "oven", "Frequent transition"),
        (("olive", "oil"), "chocolate", "Unseen transition (Zero count)"),
    ]

    methods = ["mle", "laplace", "interpolation", "stupid_backoff"]
    print(f"{'Context':<20} | {'Target':<10} | " + " | ".join(f"{m:<14}" for m in methods))
    print("-" * 88)

    for ctx, tgt, desc in scenarios:
        row = [f"{str(ctx):<20}", f"{tgt:<10}"]
        for m in methods:
            p = model.probability(tgt, ctx, method=m)
            row.append(f"{p:<14.5f}")
        print(" | ".join(row))

    # 5. Evaluate Perplexity on Held-Out Test Set
    test_file = os.path.join(os.path.dirname(__file__), "..", "data", "test_recipes.txt")
    if os.path.exists(test_file):
        print("\n[Test Set Perplexity Evaluation]")
        with open(test_file, "r", encoding="utf-8") as f:
            test_steps = [line.strip() for line in f if line.strip()]
        test_tokens = [
            corpus.tokenizer.preprocess_sentence(s, n=3, replace_unk=True)
            for s in test_steps
        ]
        for m in methods:
            ppl = model.calculate_perplexity(test_tokens, method=m)
            print(f"  Perplexity ({m:<14}): {ppl:8.2f}")

    print("\n✓ Milestone 2 completed successfully!")


if __name__ == "__main__":
    run_milestone2()
