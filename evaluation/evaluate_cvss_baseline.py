#!/usr/bin/env python3
"""
CVSS Baseline Re-Evaluation:
Re-evaluates original CVSS corpus (CVSS-C and CVSS-T, X->EN) using the exact
same evaluation pipeline (Whisper large-v3 and UTMOS) for fair, apples-to-apples comparison.

Reference: CVSS-X Short Paper Section 4.1 & Table 4
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import whisper
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from evaluation.evaluate_asr_bleu import compute_levenshtein, normalize_for_eval
from evaluation.evaluate_utmos import UTMOSEvaluator
from pipeline.utils import load_manifest, save_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xvss-x-eval")


def evaluate_cvss_manifest(
    manifest_path: Path,
    output_dir: Path,
    whisper_model_name: str = "large-v3",
    device: str = "cuda",
):
    """Evaluate original CVSS samples (target is English)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    samples = manifest.get("samples", [])

    logger.info(f"Evaluating {len(samples)} CVSS samples on {device}...")
    whisper_model = whisper.load_model(whisper_model_name, device=device)
    utmos_eval = UTMOSEvaluator(device=device)
    bleu_metric = BLEU()

    utmos_scores = []
    wers = []
    refs = []
    hyps = []

    for s in tqdm(samples, desc="Evaluating CVSS baseline"):
        audio_path = s.get("audio_path") or s.get("synth_audio_path")
        ref_en = s.get("target_text") or s.get("sentence", "")

        if not audio_path or not Path(audio_path).exists():
            continue

        # UTMOS
        try:
            u = utmos_eval.score_audio(audio_path)
            utmos_scores.append(u)
        except Exception:
            pass

        # Whisper ASR (English)
        try:
            res = whisper_model.transcribe(audio_path, language="english", task="transcribe")
            hyp_en = res.get("text", "").strip()

            ref_norm, ref_tokens = normalize_for_eval(ref_en, "en", is_unsegmented=False)
            hyp_norm, hyp_tokens = normalize_for_eval(hyp_en, "en", is_unsegmented=False)

            wer = compute_levenshtein(ref_tokens, hyp_tokens)
            wers.append(wer)

            refs.append([ref_norm])
            hyps.append(hyp_norm)
        except Exception as e:
            logger.debug(f"ASR failed for {audio_path}: {e}")

    bleu = bleu_metric.corpus_score(hyps, [[r[0] for r in refs]]).score if hyps else 0.0
    mean_wer = float(np.mean(wers)) * 100 if wers else 0.0
    mean_utmos = float(np.mean(utmos_scores)) if utmos_scores else 0.0

    result = {
        "dataset": "CVSS_original",
        "total_evaluated": len(hyps),
        "utmos_mean": round(mean_utmos, 2),
        "asr_bleu": round(bleu, 1),
        "wer_mean": round(mean_wer, 1),
    }

    out_file = output_dir / "cvss_baseline_results.json"
    save_manifest(result, out_file)
    logger.info(f"CVSS Baseline Result: UTMOS={mean_utmos:.2f}, BLEU={bleu:.1f}, WER={mean_wer:.1f}% -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate CVSS Baseline")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest file of CVSS samples")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    evaluate_cvss_manifest(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        device=args.device,
    )


if __name__ == "__main__":
    main()

