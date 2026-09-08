#!/usr/bin/env python3
"""
Step 2: Multilingual Text Translation (EN -> 28 Languages).

Translates source English transcripts into target languages using
NLLB-200-distilled-600M with batched beam search and atomic checkpointing.

Reference: CVSS-X Short Paper Section 3.1 (Text Translation)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from pipeline.utils import (
    batch_iterator,
    load_checkpoint,
    load_manifest,
    load_yaml,
    save_checkpoint,
    save_manifest,
)

logger = logging.getLogger("xvss-x")


class NLLBTranslator:
    """Batch translator using facebook/nllb-200-distilled-600M."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "eng_Latn",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        logger.info(f"Loading NLLB model: {model_name} on {device} ({torch_dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()
        logger.info("NLLB translation model successfully loaded.")

    @torch.inference_mode()
    def translate_batch(
        self,
        texts: List[str],
        tgt_lang_code: str,
        num_beams: int = 4,
        max_length: int = 256,
    ) -> List[str]:
        """Translate a batch of English texts into the target language."""
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)
        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=num_beams,
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_split(
    translator: NLLBTranslator,
    manifest_path: Path,
    output_path: Path,
    target_lang: str,
    tgt_lang_code: str,
    batch_size: int = 64,
    checkpoint_interval: int = 2000,
    max_samples: Optional[int] = None,
):
    """Translate all samples in a manifest for a single language with checkpointing."""
    manifest = load_manifest(manifest_path)
    samples = manifest.get("samples", [])
    if max_samples:
        samples = samples[:max_samples]

    checkpoint_file = output_path.parent / f"checkpoint_{manifest_path.stem}.json"
    checkpoint = load_checkpoint(checkpoint_file)
    processed_ids = set(checkpoint.get("processed_ids", []))
    processed_results = {r["id"]: r for r in checkpoint.get("results", [])}

    logger.info(
        f"Translating {len(samples):,} samples to {target_lang} ({tgt_lang_code}). "
        f"Resuming with {len(processed_ids):,} already completed."
    )

    remaining_samples = [s for s in samples if s["id"] not in processed_ids]
    results_list = list(processed_results.values())

    step = 0
    with tqdm(total=len(remaining_samples), desc=f"EN->{target_lang}") as pbar:
        for batch in batch_iterator(remaining_samples, batch_size):
            texts = [s.get("sentence", "") for s in batch]
            translations = translator.translate_batch(texts, tgt_lang_code)

            for s, tr in zip(batch, translations):
                item = dict(s)
                item["target_lang"] = target_lang
                item["target_text"] = tr
                results_list.append(item)
                processed_ids.add(s["id"])

            step += len(batch)
            pbar.update(len(batch))

            if step >= checkpoint_interval:
                save_checkpoint(checkpoint_file, list(processed_ids), results_list)
                step = 0

    # Save final manifest
    final_manifest = {
        "split": manifest.get("split", "unknown"),
        "target_lang": target_lang,
        "nllb_code": tgt_lang_code,
        "total_samples": len(results_list),
        "samples": results_list,
    }
    save_manifest(final_manifest, output_path)
    if checkpoint_file.exists():
        checkpoint_file.unlink()  # Remove checkpoint upon clean completion
    logger.info(f"Successfully saved translated manifest: {output_path} ({len(results_list):,} samples)")


def main():
    parser = argparse.ArgumentParser(description="XVSS-X Step 2: NLLB Multilingual Text Translation")
    parser.add_argument("--manifest-dir", type=str, default="data/manifests", help="Directory containing source manifests")
    parser.add_argument("--output-dir", type=str, default="data/translated", help="Base directory for translated manifests")
    parser.add_argument("--config", type=str, default="config/languages.yaml", help="Path to languages.yaml configuration")
    parser.add_argument("--languages", type=str, default="all", help="Comma-separated language codes or 'all' for 28 langs")
    parser.add_argument("--splits", type=str, default="test,dev,train", help="Comma-separated splits to translate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for translation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (for testing)")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    lang_cfg = load_yaml(args.config)["languages"]

    if args.languages.strip().lower() == "all":
        target_langs = list(lang_cfg.keys())
    else:
        target_langs = [l.strip() for l in args.languages.split(",") if l.strip()]

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    translator = NLLBTranslator(device=args.device)

    for lang in target_langs:
        if lang not in lang_cfg:
            logger.warning(f"Language '{lang}' not defined in {args.config}. Skipping.")
            continue

        tgt_code = lang_cfg[lang]["nllb_code"]
        lang_out_dir = output_dir / lang
        lang_out_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            in_file = manifest_dir / f"manifest_{split}.json"
            out_file = lang_out_dir / f"manifest_{split}_translated.json"

            if out_file.exists():
                logger.info(f"Skipping {lang}/{split} (already translated at {out_file})")
                continue

            if not in_file.exists():
                logger.warning(f"Source manifest {in_file} does not exist. Skipping.")
                continue

            translate_split(
                translator=translator,
                manifest_path=in_file,
                output_path=out_file,
                target_lang=lang,
                tgt_lang_code=tgt_code,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
            )


if __name__ == "__main__":
    main()

