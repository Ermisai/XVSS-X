# XVSS-X: A Multilingual Speech-to-Speech Translation Corpus for 28 Languages

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-Accepted-green.svg)](#citation)

This repository contains the official, reproducible pipeline and evaluation suite for **XVSS-X** (CVSS-X), a massive synthetic speech-to-speech translation (S2ST) corpus that complements and extends the original CVSS dataset by reversing the translation direction (**English $\rightarrow$ 28 target languages**).

---

## Table of Contents

- [Overview](#overview)
- [Corpus Statistics](#corpus-statistics)
- [Target Languages](#target-languages)
- [Dataset Generation Pipeline](#dataset-generation-pipeline)
- [Quality Evaluation Results](#quality-evaluation-results)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Step 1: Source Preparation (Official vs. Custom)](#2-step-1-source-preparation-official-vs-custom)
  - [3. Step 2: Multilingual Translation](#3-step-2-multilingual-translation)
  - [4. Step 3 & 4: Speech Synthesis](#4-step-3--4-speech-synthesis)
  - [5. Step 5: Evaluation Suite](#5-step-5-evaluation-suite)
- [Hugging Face Dataset Access](#hugging-face-dataset-access)
- [Roadmap: Version 2 (In Development)](#roadmap-version-2-in-development)
- [License](#license)
- [Citation](#citation)

---

## Overview

While the pioneering CVSS corpus enables translation from 21 source languages exclusively into English (many-to-one), **XVSS-X** provides one-to-many translation from English into **28 typologically diverse target languages** across 12 language families.

Combined with CVSS, XVSS-X enables:
1. **Bidirectional speech-to-speech translation** (English $\leftrightarrow$ 28 languages).
2. **Multilingual translation** between arbitrary language pairs using English as a pivot ($X \rightarrow \text{EN} \rightarrow Y$).
3. Direct evaluation of cross-lingual voice preservation in speech-to-speech tasks.

XVSS-X is provided in two complementary variants:
- **XVSS-X-C (Canonical)**: Target speech is synthesized using two fixed, high-clarity reference voices per language (one male, one female). Voice selection is conditioned on the source speaker's gender metadata (81.4% male, 18.6% female).
- **XVSS-X-T (Timbre-Transferred)**: Preserves the speaker characteristics of the original English speaker in the target speech via zero-shot cross-lingual voice cloning with [OmniVoice](https://github.com/k2-fsa/OmniVoice).

---

## Corpus Statistics

| Metric | Value |
| :--- | :--- |
| **Source Language** | English (Common Voice 17) |
| **Target Languages** | 28 |
| **Language Families** | 12 (7 macro-families) |
| **Nominal Samples per Language** | 240,192 |
| **Data Splits (per lang)** | **Train:** 222,349 \| **Dev:** 10,000 \| **Test:** 7,843 |
| **Total Parallel Speech Pairs** | 6,725,176 (Canonical) + 6,724,326 (Timbre) = **13,449,502 audio files** |
| **XVSS-X-C Duration** | $\sim$6,730 hours (avg. 3.6s / utterance) |
| **XVSS-X-T Duration** | $\sim$9,340 hours (avg. 5.0s / utterance) |
| **Total Audio Duration** | **$\sim$16,070 hours** (8$\times$ larger than CVSS) |

---

## Target Languages

XVSS-X spans 28 languages across 12 typological families:

| Family | N | Languages (ISO Codes) |
| :--- | :---: | :--- |
| **Romance** | 6 | Portuguese (`pt`), Spanish (`es`), French (`fr`), Italian (`it`), Romanian (`ro`), Catalan (`ca`) |
| **Germanic** | 5 | German (`de`), Dutch (`nl`), Swedish (`sv`), Danish (`da`), Norwegian (`no`) |
| **Slavic** | 4 | Russian (`ru`), Polish (`pl`), Czech (`cs`), Ukrainian (`uk`) |
| **CJK** | 3 | Chinese (`zh`), Japanese (`ja`), Korean (`ko`) |
| **Uralic** | 2 | Finnish (`fi`), Hungarian (`hu`) |
| **Indo-Iranian** | 2 | Hindi (`hi`), Persian (`fa`) |
| **Other** | 6 | Greek (`el`), Hebrew (`he`), Turkish (`tr`), Thai (`th`), Indonesian (`id`), Vietnamese (`vi`) |

---

## Dataset Generation Pipeline

```
Common Voice 17 EN
  (240K utterances)
        │
        ▼
┌───────────────────────────┐
│   1. Source Alignment     │ ──► Aligns CV17 recordings with CVSS metadata
└─────────────┬─────────────┘     (train: 222,349 | dev: 10,000 | test: 7,843)
              │
              ▼
┌───────────────────────────┐
│   2. Text Translation     │ ──► facebook/nllb-200-distilled-600M
└─────────────┬─────────────┘     (EN -> 28 target languages, ~17ms/sentence)
              │
              ▼
┌───────────────────────────┐
│   3. Speech Synthesis     │ ──► k2-fsa/OmniVoice (zero-shot cross-lingual TTS)
└─────────────┬─────────────┘
              │
      ┌───────┴───────────────────────┐
      ▼                               ▼
XVSS-X-C (Canonical)           XVSS-X-T (Timbre-Transferred)
2 fixed reference voices       Cloned voice from English source
(~6,730 hours)                 (~9,340 hours)
```

1. **Source Data & Alignment**: English recordings from Common Voice version 17 aligned via normalized text matching with the original CVSS corpus (recovering 91.0% of samples). The dev set is supplemented from the training split to reach exactly 10,000 samples.
2. **Text Translation**: Transcripts translated using `facebook/nllb-200-distilled-600M`, selected after extensive benchmarking across 7 translation models for its optimal quality/throughput trade-off.
3. **Speech Synthesis**: Generated with `k2-fsa/OmniVoice`, a multilingual transformer TTS engine featuring zero-shot cross-lingual synthesis without transferring source accent artifacts.

---

## Quality Evaluation Results

Evaluation performed on a stratified random sample of **200 utterances per language** from the dev set (5,600 samples per variant, determined via statistical power analysis).

### Evaluation by Language Family (CVSS-X)

| Family (N) | WER/CER (C) | WER/CER (T) | ASR-BLEU (C) | ASR-BLEU (T) | UTMOS (C) | UTMOS (T) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Romance** (6) | 5.8 | 8.2 | 90.3 | 88.0 | 3.54 | 3.27 |
| **Germanic** (5) | 9.5 | 12.2 | 85.3 | 81.1 | 3.62 | 3.27 |
| **Slavic** (4) | 7.0 | 7.7 | 86.9 | 85.9 | 3.48 | 3.17 |
| **CJK** (3)$^*$ | 5.0 | 10.2 | 75.2 | 67.9 | 3.56 | 3.20 |
| **Uralic** (2) | 10.3 | 14.3 | 84.0 | 78.3 | 3.54 | 3.21 |
| **Indo-Iranian** (2) | 22.5 | 23.0 | 63.7 | 60.0 | 3.61 | 3.20 |
| **Other** (6) | 24.8 | 24.8 | 65.0 | 64.9 | 3.53 | 3.14 |
| **Average** | **12.1** | **14.1** | **82.4** | **79.4** | **3.55** | **3.21** |

$^*$*For unsegmented CJK languages (ZH, JA, KO), Character Error Rate (CER) and character-level BLEU are reported.*

### Overall Comparison with CVSS (Re-evaluated with Same Pipeline)

| Metric | XVSS-X-C | XVSS-X-T | CVSS-C | CVSS-T |
| :--- | :---: | :---: | :---: | :---: |
| **UTMOS** (1–5) | 3.55 | 3.21 | 4.43 | 3.61 |
| **ASR-BLEU** | 82.4 | 79.4 | 94.2 | 93.8 |
| **WER/CER** (%) | 12.1 | 14.1 | 3.5 | 4.0 |
| **Speaker Similarity (ECAPA-TDNN)** | -- | **0.607** | -- | -- |

---

## Repository Structure

```
XVSS-X/
├── README.md                      # Dataset documentation and reproduction guide
├── LICENSE                        # CC-BY-NC 4.0 license
├── requirements.txt               # Unified Python dependencies
├── setup_env.sh                   # One-stop environment setup script
├── .gitignore                     # Git ignore rules
│
├── metadata/                      # Official Pre-computed Manifests
│   ├── manifest_train.json        # 222,349 source utterances
│   ├── manifest_dev.json          # 10,000 source utterances
│   └── manifest_test.json         # 7,843 source utterances
│
├── config/
│   ├── languages.yaml             # Complete metadata for all 28 target languages
│   └── pipeline.yaml              # Default hyperparameters and paths
│
├── pipeline/                      # Dataset Creation Pipeline (Clean & Modular)
│   ├── __init__.py
│   ├── 01_prepare_source.py       # CV17 extraction, CVSS alignment, and split generation
│   ├── 02_translate.py            # NLLB multilingual translation (EN -> 28 languages)
│   ├── 03_synthesize.py           # OmniVoice TTS synthesis (Canonical & Timbre modes)
│   └── utils.py                   # Data schemas, manifest I/O, checkpoints, audio helpers
│
├── evaluation/                    # Quality Evaluation Suite
│   ├── __init__.py
│   ├── sample_dev.py              # Deterministic 200 utterances/lang dev sampling (seed=42)
│   ├── evaluate_asr_bleu.py       # Whisper large-v3 ASR + SacreBLEU / chrF / CER / WER
│   ├── evaluate_utmos.py          # Neural MOS speech naturalness (UTMOS)
│   ├── evaluate_speaker_sim.py    # ECAPA-TDNN cross-lingual speaker similarity
│   ├── evaluate_cvss_baseline.py  # Re-evaluation of original CVSS baseline
│   └── generate_tables.py         # Summary report & LaTeX/Markdown table generator
│
├── scripts/                       # Reproducible Bash Runners
│   ├── 01_run_data_prep.sh        # Executes Step 1 (source data prep or load official)
│   ├── 02_run_translation.sh      # Executes Step 2 (translation)
│   ├── 03_run_synthesis_canonical.sh # Executes Step 3 (canonical synthesis)
│   ├── 04_run_synthesis_timbre.sh    # Executes Step 4 (timbre synthesis)
│   └── 05_run_evaluation.sh       # Executes Step 5 (full evaluation suite)
│
└── assets/                        # Reference voice templates
    ├── pt-male.wav
    └── pt-female.wav
```

---


## Quickstart & Reproduction

### 1. Environment Setup

Clone this repository and run the setup script:

```bash
git clone https://github.com/ErmisAI/XVSS-X.git
cd XVSS-X
./setup_env.sh
conda activate xvss-x  # or source .venv/bin/activate
```

### 2. Step 1: Source Preparation (Official vs. Custom)

You have two options to prepare the source data:

#### Option A: Use the Official Pre-computed Manifests (Recommended)
You can directly use the exact aligned splits included in this repository under `metadata/`:

```bash
# Export official manifests directly to data/manifests/
USE_OFFICIAL_MANIFESTS=true ./scripts/01_run_data_prep.sh

# Or optionally link your local Common Voice 17 audio directory:
USE_OFFICIAL_MANIFESTS=true CV17_DIR=/path/to/cv17 ./scripts/01_run_data_prep.sh
```

#### Option B: Recompute Alignment from Scratch
If you wish to re-execute text alignment from raw Common Voice v4 and v17:

```bash
CV4_DIR=/path/to/cv4 CV17_DIR=/path/to/cv17 ./scripts/01_run_data_prep.sh
```

This creates `data/manifests/manifest_{train,dev,test}.json` containing exactly 222,349 / 10,000 / 7,843 samples.

### 3. Step 2: Multilingual Translation

Translate English transcripts to all 28 target languages (or a specific subset):

```bash
# Translate all 28 languages:
./scripts/02_run_translation.sh

# Or translate specific languages (e.g., Portuguese and Spanish):
LANGS="pt,es" ./scripts/02_run_translation.sh
```

Checkpoints are saved automatically every 2,000 samples for seamless resume capability.

### 4. Step 3 & 4: Speech Synthesis

Generate speech using OmniVoice:

```bash
# 1. Canonical Variant (XVSS-X-C):
./scripts/03_run_synthesis_canonical.sh

# 2. Timbre-Transferred Variant (XVSS-X-T):
./scripts/04_run_synthesis_timbre.sh
```

Audio files are saved under `data/synthesized/en-{lang}/{split}/{variant}/{sample_id}.wav` at 24kHz.

### 5. Step 5: Evaluation Suite

Run the full evaluation pipeline (dev sampling, ASR-BLEU with Whisper large-v3, UTMOS, and ECAPA-TDNN):

```bash
./scripts/05_run_evaluation.sh
```

The resulting Markdown report and paper tables will be generated in `evaluation/evaluation_report.md`.

---

## Hugging Face Dataset Access

The complete dataset (~2.7 TB across 28 language pairs, 13.4M+ audio files) is being mirrored to the Hugging Face Hub under the account `lgris/XVSS-X`.

You will be able to load and stream subsets directly using the `datasets` library:

```python
from datasets import load_dataset

# Stream Portuguese canonical dev split
dataset = load_dataset("lgris/XVSS-X", "en-pt", split="dev", streaming=True)
sample = next(iter(dataset))
print(sample["source_text"])
print(sample["target_text"])
```


---

## Roadmap: Version 2 (In Development)

> [!NOTE]
> **We are actively working on Version 2 (v2) of the dataset.**

Key improvements planned for XVSS-X v2 include:
- **TranslateGemma-12B Integration**: Replacing NLLB-200 with Google's TranslateGemma-12B to significantly improve translation fidelity, natural phrasing, and eliminate commercial restrictions by transitioning to an **Apache 2.0** license.
- **Common Voice 26 & Spontaneous Speech 4.0**: Expanding from CV17 to Common Voice 26 and integrating Spontaneous Speech 4.0, offering far richer conversational speech, more realistic disfluencies, and greater speaker diversity.
- **End-to-End LLM Baselines**: Training and evaluating native discrete audio token speech-to-speech translation baselines.

---

## License

The XVSS-X corpus and code are licensed as follows:
- **Audio & Source Transcripts**: Inherited from Mozilla Common Voice (released under **CC0 1.0 Universal**).
- **Synthetic Translations & Speech Pairs**: Distributed under **Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)** due to the underlying `facebook/nllb-200-distilled-600M` model license.
- **Codebase & Scripts**: Released under the permissive **MIT License**.

See [LICENSE](LICENSE) for full legal text.

---

## Citation

If you use XVSS-X in your research, please cite our paper:

```bibtex
@inproceedings{gris2026cvssx,
  title={{CVSS-X: A Multilingual Speech-to-Speech Translation Corpus for 28 Languages}},
  author={Gris, Lucas Rafael Stefanel and Ferreira, Alef Iury Siqueira and de Oliveira, F. S. and da Rosa, Augusto Seben and Ferro Filho, Alexandre Costa and Galv{\~a}o Filho, Arlindo Rodrigues and Soares, Anderson da Silva},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```


