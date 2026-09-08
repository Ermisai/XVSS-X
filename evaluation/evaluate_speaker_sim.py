#!/usr/bin/env python3
"""
Speaker Similarity Evaluator (ECAPA-TDNN):
Measures cross-lingual voice cloning fidelity for XVSS-X-T by computing
cosine similarity between ECAPA-TDNN speaker embeddings of source English speech
and synthesized target speech.

Reference: CVSS-X Short Paper Section 4.1 & 4.2
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from pipeline.utils import load_manifest, save_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xvss-x-eval")


class SpeakerSimilarityEvaluator:
    """ECAPA-TDNN speaker embedding extractor and comparator."""

    def __init__(self, model_source: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = "cuda"):
        self.device = device
        logger.info(f"Loading ECAPA-TDNN model '{model_source}' on {device}...")
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source=model_source,
                run_opts={"device": device},
            )
            logger.info("ECAPA-TDNN model loaded successfully.")
        except ImportError:
            raise ImportError("speechbrain is required. Install via 'pip install speechbrain'.")

    def get_embedding(self, audio_path: str) -> torch.Tensor:
        """Load audio, resample to 16kHz, and compute speaker embedding."""
        import soundfile as sf
        wav, sr = sf.read(audio_path, dtype="float32")
        signal = torch.from_numpy(wav)
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        elif signal.ndim == 2:
            signal = signal.mean(dim=-1, keepdim=True).t()
        if sr != 16000:
            signal = torchaudio.functional.resample(signal, sr, 16000)
        signal = signal.to(self.device)
        with torch.no_grad():
            emb = self.model.encode_batch(signal)
            emb = torch.nn.functional.normalize(emb.squeeze(0), dim=-1)
        return emb

    def compute_similarity(self, ref_audio_path: str, synth_audio_path: str) -> float:
        """Compute cosine similarity between two audio files."""
        emb_ref = self.get_embedding(ref_audio_path)
        emb_synth = self.get_embedding(synth_audio_path)
        similarity = torch.sum(emb_ref * emb_synth).item()
        return float(similarity)


def evaluate_language_speaker_sim(
    evaluator: SpeakerSimilarityEvaluator,
    samples: List[Dict],
    lang: str,
) -> Dict:
    """Evaluate speaker similarity for samples of a language."""
    similarities = []
    details = []

    for s in tqdm(samples, desc=f"SpeakerSim {lang}"):
        synth_path = s.get("synth_audio_path")
        ref_path = s.get("v17_audio_path") or s.get("ref_voice_audio")

        if not synth_path or not ref_path:
            continue
        if not Path(synth_path).exists() or not Path(ref_path).exists():
            continue

        try:
            sim = evaluator.compute_similarity(ref_path, synth_path)
            similarities.append(sim)
            details.append({"sample_id": s.get("id"), "similarity": round(sim, 3)})
        except Exception as e:
            logger.debug(f"Speaker similarity failed for {s.get('id')}: {e}")

    mean_sim = float(np.mean(similarities)) if similarities else 0.0
    std_sim = float(np.std(similarities)) if similarities else 0.0

    return {
        "language": lang,
        "total_evaluated": len(similarities),
        "spk_sim_mean": round(mean_sim, 3),
        "spk_sim_std": round(std_sim, 3),
        "scores": details,
    }


def main():
    parser = argparse.ArgumentParser(description="XVSS-X Speaker Similarity Evaluation (ECAPA-TDNN)")
    parser.add_argument("--samples-file", type=str, required=True, help="JSON file with sampled evaluation items")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory for results")
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

    evaluator = SpeakerSimilarityEvaluator(device=args.device)
    results = {}

    for lang, sample_list in items.items():
        res = evaluate_language_speaker_sim(evaluator, sample_list, lang)
        results[lang] = res
        logger.info(f"[{lang}] Speaker Similarity: {res['spk_sim_mean']:.3f} ± {res['spk_sim_std']:.3f}")

    out_file = output_dir / "speaker_similarity_results.json"
    save_manifest(results, out_file)
    logger.info(f"Saved speaker similarity results to {out_file}")


if __name__ == "__main__":
    main()

