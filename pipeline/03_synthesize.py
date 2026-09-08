#!/usr/bin/env python3
"""
Step 3: Speech Synthesis with OmniVoice (Canonical & Timbre-Transferred).

Generates parallel target speech:
  - XVSS-X-C (Canonical): Conditioned on fixed male/female reference voices
    based on source speaker gender metadata (81.4% male, 18.6% female).
  - XVSS-X-T (Timbre): Conditioned on source English audio via cross-lingual
    zero-shot voice cloning. Corrupted/silent references (~3.4%) are safely skipped.

Reference: CVSS-X Short Paper Section 3.1 & 3.2
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from pipeline.utils import (
    batch_iterator,
    load_checkpoint,
    load_manifest,
    load_yaml,
    save_checkpoint,
    save_manifest,
)

logger = logging.getLogger("xvss-x")


class OmniVoiceSynthesizer:
    """TTS engine wrapper for k2-fsa/OmniVoice."""

    def __init__(
        self,
        model_name: str = "k2-fsa/OmniVoice",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        num_step: int = 48,
        guidance_scale: float = 2.5,
    ):
        self.device = device
        self.num_step = num_step
        self.guidance_scale = guidance_scale
        self.prompt_cache: Dict[str, Any] = {}

        logger.info(f"Loading OmniVoice model '{model_name}' on {device}...")
        try:
            from omnivoice.models.omnivoice import OmniVoice
            self.model = OmniVoice.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch_dtype,
            )
            self.sampling_rate = self.model.sampling_rate
            logger.info(f"OmniVoice loaded successfully! (Sampling Rate: {self.sampling_rate} Hz)")
        except ImportError:
            raise ImportError(
                "OmniVoice is not installed. Run './setup_env.sh' or 'pip install -e ./OmniVoice'."
            )

    def get_voice_prompt(self, ref_audio_path: str):
        """Extract and cache reference voice clone prompt."""
        if ref_audio_path in self.prompt_cache:
            return self.prompt_cache[ref_audio_path]

        path_obj = Path(ref_audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(path_obj),
            ref_text=None,
            preprocess_prompt=True,
        )
        self.prompt_cache[ref_audio_path] = prompt
        return prompt

    def synthesize_batch(
        self,
        texts: List[str],
        languages: List[str],
        prompts: List[Any],
    ) -> List[Optional[np.ndarray]]:
        """Synthesize batch of texts with given languages and voice prompts."""
        try:
            audios = self.model.generate(
                text=texts,
                language=languages,
                voice_clone_prompt=prompts,
                num_step=self.num_step,
                guidance_scale=self.guidance_scale,
            )
            return audios
        except Exception as e:
            logger.warning(f"Batched synthesis failed ({e}). Falling back to sequential generation...")
            # Fallback one by one
            results = []
            for t, l, p in zip(texts, languages, prompts):
                try:
                    res = self.model.generate(
                        text=[t],
                        language=[l],
                        voice_clone_prompt=[p],
                        num_step=self.num_step,
                        guidance_scale=self.guidance_scale,
                    )[0]
                    results.append(res)
                except Exception as inner_e:
                    logger.error(f"Failed individual synthesis for text '{t[:30]}...': {inner_e}")
                    results.append(None)
            return results


def synthesize_split(
    synthesizer: OmniVoiceSynthesizer,
    manifest_path: Path,
    output_audio_dir: Path,
    output_manifest_path: Path,
    target_lang: str,
    omnivoice_lang: str,
    variant: str,
    ref_male_path: Optional[str] = None,
    ref_female_path: Optional[str] = None,
    batch_size: int = 16,
    checkpoint_interval: int = 500,
    max_samples: Optional[int] = None,
):
    """Synthesize audio for a manifest with automatic checkpoint recovery."""
    manifest = load_manifest(manifest_path)
    samples = manifest.get("samples", [])
    if max_samples:
        samples = samples[:max_samples]

    checkpoint_file = output_manifest_path.parent / f"checkpoint_{manifest_path.stem}_{variant}.json"
    checkpoint = load_checkpoint(checkpoint_file)
    processed_ids = set(checkpoint.get("processed_ids", []))
    processed_results = {r["id"]: r for r in checkpoint.get("results", [])}

    logger.info(
        f"Processing {len(samples):,} samples for {target_lang} [{variant.upper()}]. "
        f"Resuming with {len(processed_ids):,} already completed."
    )

    remaining_samples = [s for s in samples if s["id"] not in processed_ids]
    results_list = list(processed_results.values())
    skipped_count = 0
    step = 0

    with tqdm(total=len(remaining_samples), desc=f"{target_lang} [{variant}]") as pbar:
        for batch in batch_iterator(remaining_samples, batch_size):
            valid_batch_samples = []
            texts = []
            langs = []
            prompts = []

            for sample in batch:
                text = sample.get("target_text", "").strip()
                if not text:
                    continue

                # Determine voice prompt reference
                if variant == "canonical":
                    gender = str(sample.get("gender", "")).lower().strip()
                    ref_audio = ref_female_path if gender == "female" else ref_male_path
                else:  # timbre
                    ref_audio = sample.get("v17_audio_path") or sample.get("v4_path")

                if not ref_audio or not Path(ref_audio).exists():
                    skipped_count += 1
                    continue

                try:
                    prompt = synthesizer.get_voice_prompt(ref_audio)
                    texts.append(text)
                    langs.append(omnivoice_lang)
                    prompts.append(prompt)
                    valid_batch_samples.append((sample, ref_audio))
                except Exception as e:
                    # Expected for ~3.4% of timbre samples with insufficient/corrupt signal
                    logger.debug(f"Skipping sample {sample.get('id')}: voice prompt error ({e})")
                    skipped_count += 1
                    continue

            if valid_batch_samples:
                generated_audios = synthesizer.synthesize_batch(texts, langs, prompts)

                for (sample, ref_audio), audio in zip(valid_batch_samples, generated_audios):
                    if audio is None or len(audio) == 0:
                        skipped_count += 1
                        continue

                    # Save WAV file
                    sample_id = sample["id"]
                    out_wav_path = output_audio_dir / f"{sample_id}.wav"
                    sf.write(str(out_wav_path), audio, synthesizer.sampling_rate)

                    duration = len(audio) / synthesizer.sampling_rate
                    item = dict(sample)
                    item["synth_audio_path"] = str(out_wav_path)
                    item["synth_duration"] = round(duration, 3)
                    item["synth_variant"] = variant
                    item["ref_voice_audio"] = str(ref_audio)

                    results_list.append(item)
                    processed_ids.add(sample_id)

            step += len(batch)
            pbar.update(len(batch))

            if step >= checkpoint_interval:
                save_checkpoint(checkpoint_file, list(processed_ids), results_list)
                step = 0

    # Save final manifest
    final_manifest = {
        "split": manifest.get("split", "unknown"),
        "target_lang": target_lang,
        "variant": variant,
        "sampling_rate": synthesizer.sampling_rate,
        "total_generated": len(results_list),
        "total_skipped": skipped_count,
        "samples": results_list,
    }
    save_manifest(final_manifest, output_manifest_path)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    logger.info(
        f"Completed {variant} synthesis for {target_lang}: "
        f"{len(results_list):,} generated, {skipped_count:,} skipped -> {output_manifest_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="XVSS-X Step 3: OmniVoice Speech Synthesis")
    parser.add_argument("--translated-dir", type=str, default="data/translated", help="Directory with translated manifests")
    parser.add_argument("--output-dir", type=str, default="data/synthesized", help="Base output directory for speech")
    parser.add_argument("--config", type=str, default="config/languages.yaml", help="Languages YAML configuration")
    parser.add_argument("--variant", type=str, required=True, choices=["canonical", "timbre"], help="Synthesis mode")
    parser.add_argument("--languages", type=str, default="all", help="Comma-separated language codes or 'all'")
    parser.add_argument("--splits", type=str, default="test,dev,train", help="Comma-separated splits to process")
    parser.add_argument("--ref-male", type=str, default="assets/pt-male.wav", help="Male reference WAV for canonical mode")
    parser.add_argument("--ref-female", type=str, default="assets/pt-female.wav", help="Female reference WAV for canonical mode")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for synthesis")
    parser.add_argument("--num-step", type=int, default=48, help="OmniVoice decoding steps (32 fast, 48 quality)")
    parser.add_argument("--guidance-scale", type=float, default=2.5, help="OmniVoice classifier-free guidance scale")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    translated_dir = Path(args.translated_dir)
    base_output_dir = Path(args.output_dir)
    lang_cfg = load_yaml(args.config)["languages"]

    if args.languages.strip().lower() == "all":
        target_langs = list(lang_cfg.keys())
    else:
        target_langs = [l.strip() for l in args.languages.split(",") if l.strip()]

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    synthesizer = OmniVoiceSynthesizer(
        device=args.device,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
    )

    for lang in target_langs:
        if lang not in lang_cfg:
            logger.warning(f"Language '{lang}' not configured. Skipping.")
            continue

        omni_name = lang_cfg[lang]["omnivoice_name"]

        for split in splits:
            trans_manifest = translated_dir / lang / f"manifest_{split}_translated.json"
            if not trans_manifest.exists():
                logger.warning(f"Missing translated manifest: {trans_manifest}. Skipping.")
                continue

            audio_out_dir = base_output_dir / f"en-{lang}" / split / args.variant
            audio_out_dir.mkdir(parents=True, exist_ok=True)
            manifest_out = base_output_dir / f"en-{lang}" / split / f"manifest_{args.variant}.json"

            if manifest_out.exists():
                logger.info(f"Skipping {lang}/{split}/{args.variant} (already completed at {manifest_out})")
                continue

            synthesize_split(
                synthesizer=synthesizer,
                manifest_path=trans_manifest,
                output_audio_dir=audio_out_dir,
                output_manifest_path=manifest_out,
                target_lang=lang,
                omnivoice_lang=omni_name,
                variant=args.variant,
                ref_male_path=args.ref_male,
                ref_female_path=args.ref_female,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
            )


if __name__ == "__main__":
    main()

