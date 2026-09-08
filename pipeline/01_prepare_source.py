#!/usr/bin/env python3
"""
Step 1: Source Data Preparation and Alignment.

Extracts English utterances from Common Voice v17 and aligns them with CVSS (CV v4)
metadata using normalized text matching, producing exact splits:
  - train: 222,349 samples
  - dev:    10,000 samples (871 original dev + 9,129 sampled from train, seed=42)
  - test:    7,843 samples
Total:     240,192 parallel source samples.

Reference: CVSS-X Short Paper Section 3.1 (Data Sources)
"""

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from pipeline.utils import load_yaml, normalize_text, save_manifest

logger = logging.getLogger("xvss-x")


def load_cv4_splits(cv4_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load Common Voice v4 TSV files (train.tsv, dev.tsv, test.tsv)."""
    splits = {}
    for split in ["train", "dev", "test"]:
        possible_paths = [
            cv4_dir / f"{split}.tsv",
            cv4_dir / "data-file" / f"{split}.tsv",
            cv4_dir / "en" / f"{split}.tsv",
        ]
        tsv_path = next((p for p in possible_paths if p.exists()), None)
        if not tsv_path:
            raise FileNotFoundError(f"Could not locate {split}.tsv in {cv4_dir}")
        logger.info(f"Loading CV v4 {split} split from {tsv_path}")
        df = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE, low_memory=False)
        splits[split] = df
    return splits


def load_cv17_metadata(cv17_dir: Path) -> pd.DataFrame:
    """Load Common Voice v17 metadata from metadata.tsv or subdirectories."""
    frames = []
    metadata_files = list(cv17_dir.rglob("metadata.tsv"))
    if not metadata_files:
        # Check for individual split TSV files
        for split in ["train", "validation", "test", "other"]:
            p = cv17_dir / f"{split}.tsv"
            if p.exists():
                metadata_files.append(p)

    if not metadata_files:
        raise FileNotFoundError(f"No metadata TSV files found in {cv17_dir}")

    for mf in metadata_files:
        logger.info(f"Loading CV v17 metadata from {mf}")
        try:
            df = pd.read_csv(mf, sep="\t", on_bad_lines="skip", low_memory=False)
            df["_source_file"] = str(mf.parent)
            frames.append(df)
        except Exception as e:
            logger.warning(f"Failed loading {mf}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} total utterances from CV v17")
    return combined


def build_sentence_index(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Index DataFrame rows by normalized sentence text."""
    index: Dict[str, List[int]] = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Indexing CV v17 sentences"):
        text = normalize_text(str(row.get("sentence", "")))
        if text:
            if text not in index:
                index[text] = []
            index[text].append(idx)
    return index


def match_and_split(
    cv4_splits: Dict[str, pd.DataFrame],
    cv17_df: pd.DataFrame,
    cv17_index: Dict[str, List[int]],
    seed: int = 42,
    target_dev_size: int = 10000,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match CV4 samples to CV17 audio and adjust splits as specified in the paper."""
    matched_by_split: Dict[str, List[Dict]] = {"train": [], "dev": [], "test": []}
    used_v17_indices: Set[int] = set()

    for split in ["test", "dev", "train"]:
        df4 = cv4_splits[split]
        for idx, row in tqdm(df4.iterrows(), total=len(df4), desc=f"Matching {split}"):
            s_norm = normalize_text(str(row.get("sentence", "")))
            if s_norm in cv17_index:
                # Pick first unused match if available, or first match
                v17_idx = None
                for candidate in cv17_index[s_norm]:
                    if candidate not in used_v17_indices:
                        v17_idx = candidate
                        used_v17_indices.add(candidate)
                        break
                if v17_idx is None:
                    v17_idx = cv17_index[s_norm][0]

                v17_row = cv17_df.iloc[v17_idx]

                # Resolve audio file path
                wav_filename = v17_row.get("filename") or v17_row.get("path")
                source_dir = Path(v17_row.get("_source_file", ""))
                audio_path = source_dir / str(wav_filename) if source_dir and wav_filename else None

                sample = {
                    "id": f"{split}_{idx}",
                    "sentence": str(row.get("sentence", "")),
                    "client_id": str(row.get("client_id", "")),
                    "gender": str(row.get("gender", "")).strip().lower() if pd.notna(row.get("gender")) else "",
                    "age": str(row.get("age", "")),
                    "accent": str(row.get("accent", "")),
                    "v4_path": str(row.get("path", "")),
                    "v17_audio_path": str(audio_path) if audio_path else "",
                    "original_split": split,
                }
                matched_by_split[split].append(sample)

    logger.info(
        f"Initial matches: train={len(matched_by_split['train'])}, "
        f"dev={len(matched_by_split['dev'])}, test={len(matched_by_split['test'])}"
    )

    # Short paper: Supplement dev split from train to reach target_dev_size (10,000)
    current_dev = matched_by_split["dev"]
    needed = target_dev_size - len(current_dev)

    if needed > 0 and len(matched_by_split["train"]) >= needed:
        logger.info(f"Supplementing dev set with {needed} samples sampled from train (seed={seed})")
        random.seed(seed)
        shuffled_train = list(matched_by_split["train"])
        random.shuffle(shuffled_train)

        dev_supplement = shuffled_train[:needed]
        final_train = shuffled_train[needed:]
        final_dev = current_dev + dev_supplement
    else:
        final_train = matched_by_split["train"]
        final_dev = current_dev

    final_test = matched_by_split["test"]

    logger.info(
        f"Final splits: train={len(final_train):,}, "
        f"dev={len(final_dev):,}, test={len(final_test):,} "
        f"(Total: {len(final_train) + len(final_dev) + len(final_test):,})"
    )

    return final_train, final_dev, final_test


def main():
    parser = argparse.ArgumentParser(description="XVSS-X Step 1: Source Data Preparation & Alignment")
    parser.add_argument("--use-official-manifests", action="store_true", help="Use pre-computed official XVSS-X manifests from repo metadata/")
    parser.add_argument("--official-metadata-dir", type=str, default="metadata", help="Directory containing official manifest_{train,dev,test}.json")
    parser.add_argument("--cv4-dir", type=str, default=None, help="Path to Common Voice v4 directory (TSVs)")
    parser.add_argument("--cv17-dir", type=str, default=None, help="Path to Common Voice v17 directory (WAVs)")
    parser.add_argument("--output-dir", type=str, default="data/manifests", help="Output directory for manifests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split sampling")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_official_manifests:
        meta_dir = Path(args.official_metadata_dir)
        logger.info(f"Using official manifests from {meta_dir}")
        for split in ["train", "dev", "test"]:
            src_file = meta_dir / f"manifest_{split}.json"
            if not src_file.exists():
                raise FileNotFoundError(f"Official manifest not found: {src_file}")
            with open(src_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", [])
            
            # If cv17_dir is provided, optionally resolve local audio path
            if args.cv17_dir:
                cv17_path = Path(args.cv17_dir)
                for s in samples:
                    if s.get("v17_wav_filename"):
                        s["audio_path"] = str(cv17_path / s["split"] / s["v17_wav_filename"])
            
            out_file = output_dir / f"manifest_{split}.json"
            manifest = {
                "split": split,
                "total_samples": len(samples),
                "samples": samples,
            }
            save_manifest(manifest, out_file)
            logger.info(f"Loaded and saved {split} manifest to {out_file} ({len(samples):,} samples)")
        return

    if not args.cv4_dir or not args.cv17_dir:
        raise ValueError("Either --use-official-manifests or both --cv4-dir and --cv17-dir must be specified.")

    cv4_dir = Path(args.cv4_dir)
    cv17_dir = Path(args.cv17_dir)

    cv4_splits = load_cv4_splits(cv4_dir)
    cv17_df = load_cv17_metadata(cv17_dir)
    cv17_index = build_sentence_index(cv17_df)

    train_samples, dev_samples, test_samples = match_and_split(
        cv4_splits, cv17_df, cv17_index, seed=args.seed
    )

    for split_name, samples in [("train", train_samples), ("dev", dev_samples), ("test", test_samples)]:
        out_file = output_dir / f"manifest_{split_name}.json"
        manifest = {
            "split": split_name,
            "total_samples": len(samples),
            "samples": samples,
        }
        save_manifest(manifest, out_file)
        logger.info(f"Saved {split_name} manifest to {out_file} ({len(samples):,} samples)")


if __name__ == "__main__":
    main()


