"""
Milestone 3 Demonstration: Create a function to generate recipe step continuations.
Develop a function for next-word prediction and text generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corpus import RecipeCorpus
from src.ngram_model import NGramModel
from src.generator import RecipeStepGenerator


def run_milestone3():
    print("=" * 70)
    print("MILESTONE 3: Next-Word Prediction & Text Generation")
    print("=" * 70)

    # 1. Setup corpus & model
    corpus = RecipeCorpus()
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "recipes_corpus.txt")
    if os.path.exists(data_file):
        corpus.load_from_text_file(data_file)
    else:
        corpus.load_seed_data()
    corpus.compile(n=3)

    model = NGramModel(n=3, smoothing="interpolation")
    model.fit(corpus.tokenized_sentences, vocab=corpus.tokenizer.vocab)
    generator = RecipeStepGenerator(model, corpus.tokenizer)

    # 2. Next-Word Prediction Demo
    prompts = [
        "Heat two tablespoons of olive oil",
        "Preheat the oven to",
        "Season the chicken generously with",
        "Bring a large pot of salted water to a",
        "Drain the cooked pasta thoroughly in a",
    ]

    print("\n[Part A: Next-Word Prediction (Top 5 Words)]")
    for prompt in prompts:
        print(f"\n  Partial Step: '{prompt}'")
        predictions = generator.predict_next_words(prompt, top_k=5)
        for rank, (word, score) in enumerate(predictions, 1):
            print(f"    {rank}. {word:<12} (score: {score:.4f})")

    # 3. Text Generation Strategies Demo
    sample_seed = "Bake in the preheated oven for"
    print(f"\n[Part B: Text Generation Strategies for '{sample_seed}']")

    # Greedy
    greedy_res = generator.generate_continuation_greedy(sample_seed, max_tokens=15)
    print(f"  • Greedy Decoding:     \"{greedy_res}\"")

    # Beam Search
    beam_res = generator.generate_continuation_beam(sample_seed, beam_width=3, max_tokens=15)
    print("  • Beam Search Top 3:")
    for i, (sent, score) in enumerate(beam_res, 1):
        print(f"      [{i}] (log-prob: {score:6.2f}) \"{sent}\"")

    # Sampling
    sample_res = generator.generate_continuation_sampling(sample_seed, max_tokens=15, temperature=0.7)
    print(f"  • Temperature Sample:  \"{sample_res}\"")

    # 4. User-Facing Step Autocompletion
    print(f"\n[Part C: User-Facing Step Autocompletion Suggestions]")
    test_inputs = [
        "In a large mixing bowl whisk together",
        "Add the minced garlic and",
        "Whisk Dijon mustard red wine vinegar and",
    ]
    for inp in test_inputs:
        print(f"\n  Input: '{inp}'")
        suggestions = generator.suggest_completions(inp, num_suggestions=3)
        for j, sugg in enumerate(suggestions, 1):
            print(f"    Suggestion {j}: \"{sugg}\"")

    print("\n✓ Milestone 3 completed successfully!")


if __name__ == "__main__":
    run_milestone3()
