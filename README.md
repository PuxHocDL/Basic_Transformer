# Basic Transformer

An educational Python project for exploring Transformer-based sequence modeling with English and Vietnamese tokenizers. The repository separates model components, dataset preparation, configuration, and training, making it a compact starting point for studying and adapting Transformer architectures.

This project is intended as a learning and experimentation codebase rather than a production-ready machine translation system. Performance benchmarks, pretrained weights, and a standardized evaluation protocol are not documented in the current repository snapshot.

## Overview

The Transformer architecture introduced in *Attention Is All You Need* (Vaswani et al., 2017) replaces recurrent sequence processing with attention mechanisms. This repository provides a code-oriented environment for investigating Transformer modeling and training with English–Vietnamese text resources.

The primary components are:

- **Model implementation:** Transformer-related modules and model construction in `model.py`.
- **Dataset processing:** Utilities for organizing model inputs and training examples in `dataset.py`.
- **Configuration:** Model and training settings maintained in `config.py`.
- **Training:** An executable training script in `train.py`, alongside `Local_train.ipynb` for notebook-based experimentation.
- **Tokenization:** Existing English and Vietnamese tokenizer files (`tokenizer_en.json` and `tokenizer_vi.json`).

## Repository Structure

```text
Basic_Transformer/
├── Local_train.ipynb     # Notebook for local experiments
├── README.md             # Project documentation
├── config.py             # Model and training configuration
├── dataset.py            # Dataset and input preparation
├── model.py              # Transformer model implementation
├── tokenizer_en.json     # English tokenizer configuration
├── tokenizer_vi.json     # Vietnamese tokenizer configuration
└── train.py              # Training entry point
```

## Transformer Background

A standard encoder–decoder Transformer is composed of token embeddings, positional information, attention layers, position-wise feed-forward networks, and an output projection. Encoder self-attention models interactions within the input sequence; a decoder, when present, uses causal self-attention and encoder–decoder attention to predict output tokens.

Scaled dot-product attention is defined as:

\[
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

Multi-head attention runs several attention operations in parallel and combines their outputs. Padding and causal masks are commonly used to control which positions can attend to one another.

> **Implementation note:** This section summarizes the original Transformer design. It should not be treated as a verified inventory of every layer or training feature in `model.py`; consult the source code for the exact implementation.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/PuxHocDL/Basic_Transformer.git
cd Basic_Transformer
```

### 2. Create a Python Environment

Python 3 and an isolated environment are recommended.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

The supplied repository listing does not include a `requirements.txt` or `pyproject.toml`. Inspect the imports in `model.py`, `dataset.py`, and `train.py`, then install the packages required by your version of the code. A typical PyTorch Transformer environment may require packages such as:

```bash
python -m pip install torch datasets tokenizers tqdm
```

This is a **suggested starting environment**, not a verified or version-pinned dependency list. Additional dependencies may be necessary, and some of the listed packages may not be needed by your local version. For reproducible experiments, create a pinned `requirements.txt` after validating the environment.

## Data and Tokenizers

The repository contains two serialized tokenizer files:

| File | Purpose |
| --- | --- |
| `tokenizer_en.json` | English text tokenization |
| `tokenizer_vi.json` | Vietnamese text tokenization |

Dataset loading and preprocessing are defined in `dataset.py`. Before training:

1. Inspect `dataset.py` to determine the input schema and expected data source.
2. Configure the data paths and model parameters in `config.py`.
3. Verify that the tokenizer vocabularies and special tokens match the training setup.
4. Ensure that the chosen training and validation splits do not overlap.

The repository snapshot does not identify a downloadable dataset, so this README deliberately does not prescribe a data file name or directory that may not exist.

## Training

Review `config.py` and the dataset setup before launching a run. The repository includes a training script:

```bash
python train.py
```

Alternatively, open `Local_train.ipynb` in JupyterLab or VS Code to inspect and execute the notebook workflow:

```bash
python -m pip install jupyterlab
jupyter lab Local_train.ipynb
```

**Before running a longer experiment**, check the source code for its supported device selection, batch size, number of epochs, learning rate, output directories, checkpoint behavior, and validation routine. The precise values and command-line arguments have not been independently verified from the repository files.

## Extending the Project

This structure is suitable for controlled learning experiments, for example:

- Comparing different attention-head counts or model dimensions, where supported by `model.py` and `config.py`.
- Evaluating the effects of tokenization and sequence length on bilingual text modeling.
- Adding an explicit held-out evaluation script and task-appropriate metrics.
- Recording seeds, package versions, model configurations, and checkpoints for reproducibility.

These are suggested directions for further work, not claims about experiments already completed.

## Reproducibility and Limitations

The current repository listing does not include a pinned dependency manifest, benchmark table, pretrained model weights, or standalone inference/evaluation script. Accordingly, no accuracy, BLEU score, model size, training cost, or hardware requirement is claimed here.

For research use, document the dataset provenance and licensing, preprocessing decisions, configuration values, random seeds, hardware, software versions, and evaluation protocol alongside any reported results.

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems (NeurIPS)*.

## Maintainer

[Nguyen Xuan Phuc (PuxHocDL)](https://github.com/PuxHocDL)

## License

No license file was shown in the supplied repository listing. Until a license is added, do not assume that the source is licensed for unrestricted redistribution or modification.
