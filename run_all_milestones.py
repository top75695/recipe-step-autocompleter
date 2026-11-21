"""
Master Runner for All 4 Project Milestones.
Executes and validates Milestones 1 through 4 in sequence.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from milestones.milestone1_corpus import run_milestone1
from milestones.milestone2_ngram import run_milestone2
from milestones.milestone3_generator import run_milestone3
from milestones.milestone4_cli import run_milestone4


def main():
    print("\n" + "=" * 76)
    print("      RECIPE STEP AUTOCOMPLETER - FULL MILESTONE VERIFICATION SUITE")
    print("=" * 76 + "\n")

    start_time = time.time()

    # Milestone 1
    run_milestone1()
    print("\n")

    # Milestone 2
    run_milestone2()
    print("\n")

    # Milestone 3
    run_milestone3()
    print("\n")

    # Milestone 4
    run_milestone4()
    print("\n")

    elapsed = round(time.time() - start_time, 2)
    print("=" * 76)
    print(f"🎉 ALL 4 MILESTONES VERIFIED AND PASSED SUCCESSFULLY in {elapsed}s!")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
