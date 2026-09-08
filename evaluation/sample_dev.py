#!/usr/bin/env python3
"""
Evaluation Sample Selector:
Extracts a stratified random sample of 200 utterances per language from the dev set
(5,600 samples per variant) using a deterministic seed (seed=42).

Reference: CVSS-X Short Paper Section 4.1 (Evaluation Setup)
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from pipeline.utils import load_manifest, load_yaml, save_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xvss-x-eval")


def sample_evaluation_set(
    data_dir: Path,
    output_dir: Path,
    languages: List[str],
    variant: str,
    sample_size: int = 200,
    seed: int = 42,
):
    """Deterministically sample evaluation samples for each language."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_eval_samples: Dict[str, List[Dict]] = {}

    for lang in languages:
        manifest_path = data_dir / f"en-{lang}" / "dev" / f"manifest_{variant}.json"
        if not manifest_path.exists():
            logger.warning(f"Manifest not found for {lang} ({manifest_path}). Skipping.")
            continue

        manifest = load_manifest(manifest_path)
        samples = manifest.get("samples", [])

        # Filter only samples with valid audio files
        valid_samples = [
            s for s in samples
            if "synth_audio_path" in s and Path(s["synth_audio_path"]).exists()
        ]

        if len(valid_samples) < sample_size:
            logger.warning(
                f"{lang}: Only {len(valid_samples)} valid samples available "
                f"(requested {sample_size}). Taking all."
            )
            sampled = valid_samples
        else:
            # Deterministic shuffle with per-language salt
            rng = random.Random(seed + hash(lang) % 100000)
            sampled = rng.sample(valid_samples, sample_size)

        all_eval_samples[lang] = sampled
        out_file = output_dir / f"eval_samples_{lang}_{variant}.json"
        save_manifest({"language": lang, "variant": variant, "samples": sampled}, out_file)
        logger.info(f"{lang} [{variant}]: Selected {len(sampled)} samples -> {out_file}")

    combined_out = output_dir / f"eval_samples_all_{variant}.json"
    save_manifest(all_eval_samples, combined_out)
    total_selected = sum(len(v) for v in all_eval_samples.values())
    logger.info(f"Total evaluation set [{variant}]: {total_selected:,} utterances -> {combined_out}")


def main():
    parser = argparse.ArgumentParser(description="XVSS-X Evaluation Sampling")
    parser.add_argument("--data-dir", type=str, default="data/synthesized", help="Directory with synthesized data")
    parser.add_argument("--output-dir", type=str, default="evaluation/samples", help="Output directory for samples")
    parser.add_argument("--config", type=str, default="config/languages.yaml", help="Languages configuration")
    parser.add_argument("--variant", type=str, required=True, choices=["canonical", "timbre"], help="Synthesis variant")
    parser.add_argument("--sample-size", type=int, default=200, help="Samples per language")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    lang_cfg = load_yaml(args.config)["languages"]
    sample_evaluation_set(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        languages=list(lang_cfg.keys()),
        variant=args.variant,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

