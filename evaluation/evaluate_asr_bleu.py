#!/usr/bin/env python3
"""
ASR-BLEU & Intelligibility Evaluator:
Transcribes synthesized speech with Whisper large-v3 and computes:
  - ASR-BLEU (SacreBLEU)
  - chrF / chrF++
  - WER (Word Error Rate) for alphabetic languages
  - CER (Character Error Rate) and character-level BLEU for CJK & unsegmented languages

Reference: CVSS-X Short Paper Section 4.1 & 4.2
"""

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import whisper
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

from pipeline.utils import load_manifest, load_yaml, save_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xvss-x-eval")


def compute_levenshtein(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """Compute error rate (WER or CER) between reference and hypothesis tokens."""
    if not ref_tokens:
        return 1.0 if hyp_tokens else 0.0

    d = np.zeros((len(ref_tokens) + 1, len(hyp_tokens) + 1), dtype=np.int32)
    for i in range(len(ref_tokens) + 1):
        d[i, 0] = i
    for j in range(len(hyp_tokens) + 1):
        d[0, j] = j

    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = min(
                d[i - 1, j] + 1,      # deletion
                d[i, j - 1] + 1,      # insertion
                d[i - 1, j - 1] + cost  # substitution
            )

    return float(d[len(ref_tokens), len(hyp_tokens)]) / len(ref_tokens)


def normalize_for_eval(text: str, lang: str, is_unsegmented: bool = False) -> Tuple[str, List[str]]:
    """Normalize text and return both normalized string and token list for WER/CER."""
    if not text:
        return "", []

    text = unicodedata.normalize("NFC", text).strip()
    # Remove punctuation
    text = re.sub(r'[.,!?;:\-\"\'\(\)\[\]{}，。！？：；、「」『』【】《》\n\r\t]', " ", text)

    if is_unsegmented:
        # For CJK & Thai: remove all spaces, token list is list of characters
        text_clean = re.sub(r"\s+", "", text)
        tokens = list(text_clean)
        bleu_text = " ".join(tokens)  # Char-separated for BLEU
        return bleu_text, tokens
    else:
        # For standard languages: lower-case, collapse whitespace
        text_clean = re.sub(r"\s+", " ", text).strip().lower()
        tokens = text_clean.split()
        return text_clean, tokens


class ASREvaluator:
    """Transcription and translation fidelity evaluator using Whisper large-v3."""

    def __init__(self, model_name: str = "large-v3", device: str = "cuda"):
        self.device = device
        logger.info(f"Loading Whisper {model_name} on {device}...")
        self.model = whisper.load_model(model_name, device=device)
        self.bleu_metric = BLEU()
        self.chrf_metric = CHRF()
        logger.info("Whisper model loaded.")

    def transcribe(self, audio_path: str, language: str) -> str:
        """Transcribe audio file in target language."""
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            temperature=0.0,
            beam_size=5,
        )
        return result.get("text", "").strip()


def evaluate_language_asr(
    evaluator: ASREvaluator,
    samples: List[Dict],
    target_lang: str,
    whisper_lang_code: str,
    is_unsegmented: bool = False,
) -> Dict:
    """Evaluate ASR-BLEU, chrF, and WER/CER for a list of samples."""
    references = []
    hypotheses = []
    error_rates = []
    detailed_results = []

    for s in tqdm(samples, desc=f"ASR-BLEU {target_lang}"):
        audio_path = s.get("synth_audio_path")
        ref_text = s.get("target_text", "")

        if not audio_path or not Path(audio_path).exists():
            continue

        hyp_text = evaluator.transcribe(audio_path, whisper_lang_code)

        ref_norm, ref_tokens = normalize_for_eval(ref_text, target_lang, is_unsegmented)
        hyp_norm, hyp_tokens = normalize_for_eval(hyp_text, target_lang, is_unsegmented)

        err_rate = compute_levenshtein(ref_tokens, hyp_tokens)
        error_rates.append(err_rate)

        references.append([ref_norm])
        hypotheses.append(hyp_norm)

        detailed_results.append({
            "sample_id": s.get("id"),
            "reference": ref_text,
            "hypothesis": hyp_text,
            "error_rate": round(err_rate * 100, 2),
        })

    # Compute corpus-level SacreBLEU and chrF
    if hypotheses and references:
        # Transpose references for SacreBLEU: List[List[str]] where outer list is num_refs
        refs_transposed = [[r[0] for r in references]]
        bleu_score = evaluator.bleu_metric.corpus_score(hypotheses, refs_transposed).score
        chrf_score = evaluator.chrf_metric.corpus_score(hypotheses, refs_transposed).score
        mean_err = float(np.mean(error_rates)) * 100 if error_rates else 0.0
    else:
        bleu_score, chrf_score, mean_err = 0.0, 0.0, 0.0

    metric_name = "CER" if is_unsegmented else "WER"

    return {
        "language": target_lang,
        "total_evaluated": len(hypotheses),
        "asr_bleu": round(bleu_score, 2),
        "chrf": round(chrf_score, 2),
        f"{metric_name.lower()}_mean": round(mean_err, 2),
        "error_metric": metric_name,
        "samples": detailed_results,
    }


def main():
    parser = argparse.ArgumentParser(description="XVSS-X ASR-BLEU & Error Rate Evaluation")
    parser.add_argument("--samples-file", type=str, required=True, help="JSON file with sampled evaluation items")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory for evaluation results")
    parser.add_argument("--config", type=str, default="config/languages.yaml", help="Languages configuration")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lang_cfg = load_yaml(args.config)["languages"]

    data = load_manifest(args.samples_file)
    lang_key = data.get("target_lang") or data.get("language")
    if "samples" in data and lang_key:
        items = {lang_key: data["samples"]}
    elif isinstance(data, dict):
        items = data
    else:
        raise ValueError("Unsupported format for samples file")

    evaluator = ASREvaluator(device=args.device)
    results = {}

    for lang, sample_list in items.items():
        if lang not in lang_cfg:
            continue
        cfg = lang_cfg[lang]
        res = evaluate_language_asr(
            evaluator=evaluator,
            samples=sample_list,
            target_lang=lang,
            whisper_lang_code=cfg["whisper_code"],
            is_unsegmented=cfg.get("unsegmented", False),
        )
        results[lang] = res
        metric = res["error_metric"]
        err_val = res[f"{metric.lower()}_mean"]
        logger.info(
            f"[{lang}] ASR-BLEU: {res['asr_bleu']:.2f} | chrF: {res['chrf']:.2f} | {metric}: {err_val:.1f}%"
        )

    out_file = output_dir / "asr_bleu_results.json"
    save_manifest(results, out_file)
    logger.info(f"Saved ASR evaluation results to {out_file}")


if __name__ == "__main__":
    main()
