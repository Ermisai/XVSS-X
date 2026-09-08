#!/usr/bin/env python3
"""
Speech Naturalness Evaluator (UTMOS):
Predicts Mean Opinion Score (1-5 scale) using UTMOS (Saeki et al., 2022).

Reference: CVSS-X Short Paper Section 4.1 & 4.2
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
from tqdm import tqdm

from pipeline.utils import load_manifest, load_yaml, save_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xvss-x-eval")


class UTMOSEvaluator:
    """Neural MOS predictor wrapper."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.info("Loading UTMOS (utmos22_strong) model from torch.hub...")
        self.model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        ).to(device)
        self.model.eval()
        logger.info("UTMOS model loaded.")

    def score_audio(self, audio_path: str) -> float:
        """Calculate UTMOS score for a single audio file (16kHz)."""
        wav, sr = librosa.load(audio_path, sr=16000)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(wav_tensor, sr).item()
        return float(score)


def evaluate_language_utmos(evaluator: UTMOSEvaluator, samples: List[Dict], lang: str) -> Dict:
    """Evaluate UTMOS scores for samples of a specific language."""
    scores = []
    details = []

    for s in tqdm(samples, desc=f"UTMOS {lang}"):
        audio_path = s.get("synth_audio_path")
        if not audio_path or not Path(audio_path).exists():
            continue

        try:
            score = evaluator.score_audio(audio_path)
            scores.append(score)
            details.append({"sample_id": s.get("id"), "utmos": round(score, 3)})
        except Exception as e:
            logger.warning(f"Failed UTMOS for {audio_path}: {e}")

    mean_score = float(np.mean(scores)) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0

    return {
        "language": lang,
        "total_evaluated": len(scores),
        "utmos_mean": round(mean_score, 3),
        "utmos_std": round(std_score, 3),
        "scores": details,
    }


def main():
    parser = argparse.ArgumentParser(description="XVSS-X Speech Naturalness (UTMOS) Evaluation")
    parser.add_argument("--samples-file", type=str, required=True, help="JSON file with sampled evaluation items")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory for evaluation results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_manifest(args.samples_file)
    lang_key = data.get("target_lang") or data.get("language")
    if "samples" in data and lang_key:
        items = {lang_key: data["samples"]}
    elif isinstance(data, dict):
        items = data
    else:
        raise ValueError("Unsupported format for samples file")

    evaluator = UTMOSEvaluator(device=args.device)
    results = {}

    for lang, sample_list in items.items():
        res = evaluate_language_utmos(evaluator, sample_list, lang)
        results[lang] = res
        logger.info(f"[{lang}] UTMOS: {res['utmos_mean']:.2f} ± {res['utmos_std']:.2f}")

    out_file = output_dir / "utmos_results.json"
    save_manifest(results, out_file)
    logger.info(f"Saved UTMOS results to {out_file}")


if __name__ == "__main__":
    main()

