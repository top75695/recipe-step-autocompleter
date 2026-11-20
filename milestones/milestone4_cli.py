"""
Milestone 4 Demonstration: Build a CLI for users to input partial steps and get suggestions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cli import RecipeAutocompleterCLI


def run_milestone4():
    print("=" * 70)
    print("MILESTONE 4: CLI for Recipe Step Autocompletion")
    print("=" * 70)

    # 1. Initialize CLI
    cli = RecipeAutocompleterCLI(n=3, smoothing="interpolation")

    print("\n[Simulating CLI User Queries]")
    test_queries = [
        "Heat olive oil in a",
        "Preheat the oven to",
        "Bake in the preheated oven for",
        "Season with kosher salt and",
    ]

    for q in test_queries:
        cli.autocomplete_and_display(q, top_k_words=4, num_phrases=3)

    print("\n[Perplexity & Evaluation Matrix]")
    cli.evaluate_smoothing_methods()

    print("\n✓ Milestone 4 completed successfully!")


if __name__ == "__main__":
    run_milestone4()
