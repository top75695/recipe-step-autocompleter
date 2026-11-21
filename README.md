# 🍳 Recipe Step Autocompleter

> **askThili Drill Project: Build a Next-Word Predictor from Scratch**  
> *Category: Travel & Hospitality | Runs on your laptop | Command line*

A command-line tool that assists home cooks in autocompleting recipe instructions, ensuring recipes can be followed smoothly and intuitively without missing critical steps. Built completely from scratch in pure Python using statistical n-gram language modeling, Maximum Likelihood Estimation (MLE), and advanced smoothing techniques.

---

## 📋 Project Milestones & Verification

| Milestone | Description | Status | Verification Module |
| :--- | :--- | :---: | :--- |
| **Milestone 1** | **Compile a corpus of cooking recipe instructions.**<br>• Tokenize and preprocess text corpus for n-gram analysis.<br>• Handle culinary terms, measurements, fractions, and sentence markers (`<s>`, `</s>`, `<unk>`). | ✅ **Done** | `milestones/milestone1_corpus.py` |
| **Milestone 2** | **Implement n-gram counting & maximum-likelihood probabilities with smoothing.**<br>• Implement multi-order n-gram frequency distributions.<br>• Unsmoothed MLE, Laplace (Add-1) / Lidstone (Add-k), Linear Interpolation, and Stupid Backoff.<br>• Test-set perplexity evaluation matrix. | ✅ **Done** | `milestones/milestone2_ngram.py` |
| **Milestone 3** | **Create a function to generate recipe step continuations.**<br>• Next-word top-$k$ prediction with probability distributions.<br>• Step continuation via Greedy decoding, Beam Search ($k=3$), and Temperature-scaled sampling. | ✅ **Done** | `milestones/milestone3_generator.py` |
| **Milestone 4** | **Build a CLI for users to input partial steps and get suggestions.**<br>• Interactive terminal mode with real-time autocompletion.<br>• One-shot query mode, prediction mode, evaluation mode, and dynamic model reconfiguration. | ✅ **Done** | `src/cli.py` / `main.py` |

---

## 🚀 Quick Start

### 1. Run the Interactive CLI
Launch the interactive autocompleter terminal:
```bash
python main.py
```
Or directly via the CLI module:
```bash
python src/cli.py
```

### 2. Run a One-Shot Query
You can pass any partial cooking instruction directly as an argument:
```bash
python main.py "Heat olive oil in a"
python main.py "Preheat the oven to"
python main.py "Season with kosher salt and"
```

### 3. Predict Top Next Words
```bash
python main.py --predict "Bake in the preheated oven for"
```

### 4. Evaluate Smoothing Techniques (Perplexity)
```bash
python main.py --evaluate
```

### 5. Run Full Milestone Verification Suite
Execute all 4 milestone demonstrations end-to-end:
```bash
python run_all_milestones.py
```

### 6. Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 🧮 Mathematical Formulations

### 1. Maximum Likelihood Estimation (MLE)
For an $n$-gram $(w_1, w_2, \dots, w_n)$, the context is $w_1^{n-1} = (w_1, \dots, w_{n-1})$:
$$P_{MLE}(w_n \mid w_1^{n-1}) = \frac{C(w_1, \dots, w_n)}{C(w_1, \dots, w_{n-1})}$$

### 2. Lidstone / Laplace (Add-$k$) Smoothing
Prevents zero probability for unseen words in a known context:
$$P_{add-k}(w_n \mid w_1^{n-1}) = \frac{C(w_1, \dots, w_n) + k}{C(w_1, \dots, w_{n-1}) + k \cdot |V|}$$

### 3. Jelinek-Mercer Linear Interpolation
Combines unigram, bigram, and trigram maximum-likelihood estimates weighted by $\lambda_1, \lambda_2, \lambda_3$ where $\sum \lambda_i = 1$:
$$\hat{P}(w_3 \mid w_1, w_2) = \lambda_3 P_{MLE}(w_3 \mid w_1, w_2) + \lambda_2 P_{MLE}(w_3 \mid w_2) + \lambda_1 P_{MLE}(w_3)$$

### 4. Stupid Backoff (Brants et al., 2007)
A fast, scalable backoff scheme using penalty factor $\alpha = 0.4$:
$$S(w_n \mid w_1^{n-1}) = \begin{cases} \frac{C(w_1, \dots, w_n)}{C(w_1, \dots, w_{n-1})} & \text{if } C(w_1, \dots, w_n) > 0 \\ \alpha \cdot S(w_n \mid w_2^{n-1}) & \text{otherwise} \end{cases}$$
Base case (unigram): $S(w) = \frac{C(w)}{N_{total}}$.

### 5. Test Perplexity (PPL)
Quantitative evaluation on held-out test recipes (lower is better):
$$\text{PPL}(W) = \exp\left(-\frac{1}{M} \sum_{i=1}^M \ln P(w_i \mid w_{i-n+1}^{i-1})\right)$$

**Empirical Results on Test Recipes:**
- Unsmoothed MLE: **2291.99** (severe penalty on zero counts)
- Laplace (Add-1): **334.91**
- Linear Interpolation: **17.19**
- Stupid Backoff: **13.00**

---

## 📁 Repository Structure

```
recipe-step-autocompleter/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Dependencies (standard library + pytest)
├── main.py                        # Root CLI entry point
├── run_all_milestones.py          # Master milestone test and verification runner
├── data/
│   ├── recipes_corpus.txt         # Recipe step corpus (baking, frying, boiling, etc.)
│   └── test_recipes.txt           # Held-out recipe steps for perplexity evaluation
├── src/
│   ├── __init__.py
│   ├── preprocessor.py            # Milestone 1: Tokenizer, padding & detokenizer
│   ├── corpus.py                  # Milestone 1: Corpus compilation & loader
│   ├── ngram_model.py             # Milestone 2: N-gram counting, MLE & smoothing
│   ├── generator.py               # Milestone 3: Next-word prediction & beam generation
│   └── cli.py                     # Milestone 4: Interactive and command-line UI
├── milestones/
│   ├── __init__.py
│   ├── milestone1_corpus.py       # Milestone 1 demo script
│   ├── milestone2_ngram.py        # Milestone 2 demo script
│   ├── milestone3_generator.py    # Milestone 3 demo script
│   └── milestone4_cli.py          # Milestone 4 demo script
└── tests/
    ├── __init__.py
    ├── test_preprocessor.py       # Tests for tokenizer & padding
    ├── test_ngram_model.py        # Tests for n-gram counting, MLE & smoothing
    └── test_generator.py          # Tests for prediction & autocompletion
```

---

## 💻 CLI Commands & Controls

Inside the interactive prompt (`python main.py`):
- Any text: e.g. `Heat olive oil in a` -> provides top next words + top 3 completed sentences
- `:stats` -> displays corpus token statistics and model configuration
- `:model <n> <method>` -> dynamically changes n-gram order ($n$) and smoothing method (`interpolation`, `stupid_backoff`, `laplace`, `mle`)
- `:test` -> runs automated recipe prompt suite
- `:exit` / `:quit` -> exits the application
