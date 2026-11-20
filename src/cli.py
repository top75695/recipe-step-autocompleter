"""
Command-Line Interface (CLI) for Recipe Step Autocompleter.
Milestone 4: Build a CLI for users to input partial steps and get suggestions.
"""

import sys
import os
import argparse
from typing import List, Tuple, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import RecipeTokenizer
from src.corpus import RecipeCorpus
from src.ngram_model import NGramModel
from src.generator import RecipeStepGenerator


# Color codes for terminal UI
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def format_prob_bar(prob: float, bar_len: int = 15) -> str:
    """Formats a visual probability progress bar."""
    filled = int(round(prob * bar_len))
    filled = max(0, min(bar_len, filled))
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"{bar} {prob * 100:5.1f}%"


class RecipeAutocompleterCLI:
    """CLI application orchestrating the recipe autocompleter experience."""

    def __init__(self, corpus_path: Optional[str] = None, n: int = 3, smoothing: str = "interpolation"):
        self.n = n
        self.smoothing = smoothing
        self.tokenizer = RecipeTokenizer(lowercase=True)
        self.corpus = RecipeCorpus(self.tokenizer)

        # Load corpus
        if corpus_path and os.path.exists(corpus_path):
            self.corpus.load_from_text_file(corpus_path)
        else:
            default_path = os.path.join(os.path.dirname(__file__), "..", "data", "recipes_corpus.txt")
            if os.path.exists(default_path):
                self.corpus.load_from_text_file(default_path)
            else:
                self.corpus.load_seed_data()

        self.corpus.compile(n=self.n)

        # Train model
        self.model = NGramModel(n=self.n, smoothing=self.smoothing)
        self.model.fit(self.corpus.tokenized_sentences, vocab=self.tokenizer.vocab)

        self.generator = RecipeStepGenerator(self.model, self.tokenizer)

    def print_banner(self) -> None:
        """Prints the welcome banner and instructions."""
        print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
        print(f"{BOLD}{GREEN}       🍳 RECIPE STEP AUTOCOMPLETER (askThili Drill Project) 🍳        {RESET}")
        print(f"{BOLD}{CYAN}======================================================================{RESET}")
        print(f"{DIM}AI-powered next-word predictor & step autocompleter for home cooks.{RESET}")
        print(f"Model: {BOLD}{self.n}-gram{RESET} | Smoothing: {BOLD}{self.smoothing.upper()}{RESET} | Recipes in corpus: {BOLD}{len(self.corpus.raw_sentences)}{RESET} | Vocab: {BOLD}{len(self.tokenizer.vocab)}{RESET}\n")
        print(f"{YELLOW}How to use:{RESET}")
        print("  • Type any partial cooking instruction (e.g., 'Heat olive oil in a')")
        print("  • Press Enter to see top next words and full recipe step completions")
        print("  • Commands: ':stats', ':model <n> <smoothing>', ':test', ':exit'\n")

    def autocomplete_and_display(self, user_input: str, top_k_words: int = 5, num_phrases: int = 3) -> None:
        """Processes partial step input, predicts next words and full step continuations."""
        cleaned = user_input.strip()
        if not cleaned:
            return

        print(f"\n{BOLD}Input Partial Step:{RESET} {CYAN}\"{cleaned}\"{RESET}")

        # 1. Top Next Words
        print(f"\n{BOLD}🎯 Top Next-Word Predictions:{RESET}")
        predictions = self.generator.predict_next_words(cleaned, top_k=top_k_words)
        if predictions:
            # Re-normalize displayed distribution among top candidates for intuitive viewing
            sum_probs = sum(p for _, p in predictions) or 1.0
            for rank, (word, raw_prob) in enumerate(predictions, 1):
                norm_prob = raw_prob / sum_probs
                bar = format_prob_bar(norm_prob)
                print(f"  {rank}. {BOLD}{word:<15}{RESET} [{bar}] (score: {raw_prob:.4f})")
        else:
            print(f"  {DIM}No specific next word found in vocabulary.{RESET}")

        # 2. Suggested Full Recipe Continuations
        print(f"\n{BOLD}✨ Suggested Complete Recipe Steps:{RESET}")
        completions = self.generator.suggest_completions(cleaned, num_suggestions=num_phrases)
        for i, phrase in enumerate(completions, 1):
            print(f"  [{i}] {GREEN}{phrase}{RESET}")
        print()

    def run_interactive(self) -> None:
        """Starts the interactive prompt loop."""
        self.print_banner()

        sample_prompts = [
            "Heat olive oil in a",
            "Bake in the preheated oven for",
            "Season generously with",
            "Bring a large pot of",
            "Mince four cloves of",
            "Whisk the eggs and",
        ]
        print(f"{DIM}Quick test ideas: {', '.join(repr(p) for p in sample_prompts[:3])}{RESET}\n")

        while True:
            try:
                user_input = input(f"{BOLD}{MAGENTA}Enter partial step > {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{CYAN}Happy cooking! Goodbye! 👋{RESET}")
                break

            if not user_input:
                continue

            if user_input.lower() in (":exit", ":quit", "exit", "quit", "q"):
                print(f"\n{CYAN}Happy cooking! Goodbye! 👋{RESET}")
                break

            elif user_input.lower() == ":stats":
                stats = self.corpus.get_stats()
                print(f"\n{BOLD}Corpus & Model Statistics:{RESET}")
                print(f"  • Total Steps: {stats['total_recipe_steps']}")
                print(f"  • Total Tokens: {stats['total_tokens']}")
                print(f"  • Vocabulary Size: {stats['vocab_size']}")
                print(f"  • Avg Tokens/Step: {stats['avg_tokens_per_step']}")
                print(f"  • N-gram Order: {self.n}")
                print(f"  • Smoothing Algorithm: {self.smoothing}\n")

            elif user_input.lower().startswith(":model"):
                parts = user_input.split()
                if len(parts) >= 2:
                    try:
                        new_n = int(parts[1])
                        new_smoothing = parts[2].lower() if len(parts) >= 3 else self.smoothing
                        print(f"{YELLOW}Reconfiguring model to n={new_n}, smoothing={new_smoothing}...{RESET}")
                        self.n = new_n
                        self.smoothing = new_smoothing
                        self.corpus.compile(n=self.n)
                        self.model = NGramModel(n=self.n, smoothing=self.smoothing)
                        self.model.fit(self.corpus.tokenized_sentences, vocab=self.tokenizer.vocab)
                        self.generator = RecipeStepGenerator(self.model, self.tokenizer)
                        print(f"{GREEN}✓ Model reconfigured successfully!{RESET}\n")
                    except Exception as e:
                        print(f"{YELLOW}Error reconfiguring model: {e}{RESET}\n")
                else:
                    print(f"{YELLOW}Usage: :model <n> [smoothing: mle|laplace|interpolation|stupid_backoff]{RESET}\n")

            elif user_input.lower() == ":test":
                print(f"\n{YELLOW}Running automated test suite across sample cooking steps...{RESET}")
                for prompt in sample_prompts:
                    self.autocomplete_and_display(prompt, top_k_words=3, num_phrases=2)

            elif user_input.lower() in (":help", "help"):
                print(f"\n{BOLD}Commands:{RESET}")
                print("  :stats            Show corpus and model statistics")
                print("  :model <n> <mth>  Switch order (e.g. 2, 3, 4) and smoothing (mle, laplace, interpolation, stupid_backoff)")
                print("  :test             Run through built-in sample recipe prompts")
                print("  :exit             Quit the program\n")

            else:
                self.autocomplete_and_display(user_input)

    def evaluate_smoothing_methods(self, test_filepath: Optional[str] = None) -> None:
        """Evaluates and compares perplexity of all smoothing techniques on test recipe steps."""
        test_path = test_filepath or os.path.join(os.path.dirname(__file__), "..", "data", "test_recipes.txt")
        if not os.path.exists(test_path):
            print(f"Test file not found: {test_path}")
            return

        with open(test_path, "r", encoding="utf-8") as f:
            test_steps = [line.strip() for line in f if line.strip()]

        test_tokenized = [
            self.tokenizer.preprocess_sentence(s, n=self.n, replace_unk=True)
            for s in test_steps
        ]

        methods = ["mle", "laplace", "interpolation", "stupid_backoff"]
        print(f"\n{BOLD}{CYAN}=== Smoothing Evaluation: Test Set Perplexity ({len(test_steps)} steps) ==={RESET}")
        print(f"{'Method':<20} | {'Perplexity':<15} | {'Notes'}")
        print("-" * 65)

        for m in methods:
            ppl = self.model.calculate_perplexity(test_tokenized, method=m)
            notes = "Lower is better"
            if m == "mle" and ppl > 10000:
                notes = "Suffers from zero-probability sparse n-grams"
            elif m == "interpolation":
                notes = "Combines unigram, bigram & trigram weights"
            elif m == "stupid_backoff":
                notes = "Fast backoff score (Brants et al.)"
            elif m == "laplace":
                notes = "Add-1 smoothing across vocabulary"
            print(f"{m.upper():<20} | {ppl:<15.2f} | {notes}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Recipe Step Autocompleter: Next-word prediction & step completion from scratch."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Optional partial recipe step to autocomplete in one shot (e.g., 'Heat olive oil in a')"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to recipe corpus text file (defaults to data/recipes_corpus.txt)"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=3,
        help="N-gram order (default: 3 for trigram)"
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        default="interpolation",
        choices=["mle", "laplace", "interpolation", "stupid_backoff"],
        help="Smoothing technique (default: interpolation)"
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Predict top next words for given partial step"
    )
    parser.add_argument(
        "--generate",
        type=str,
        default=None,
        help="Generate full step continuation for given partial step"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run perplexity evaluation across all smoothing methods on test set"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal prompt (default if no argument supplied)"
    )

    args = parser.parse_args()
    cli = RecipeAutocompleterCLI(corpus_path=args.corpus, n=args.n, smoothing=args.smoothing)

    if args.evaluate:
        cli.evaluate_smoothing_methods()
    elif args.predict:
        cli.autocomplete_and_display(args.predict, top_k_words=5, num_phrases=0)
    elif args.generate:
        cli.autocomplete_and_display(args.generate, top_k_words=0, num_phrases=3)
    elif args.query:
        cli.autocomplete_and_display(args.query)
    else:
        # Default to interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()
