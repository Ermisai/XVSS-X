"""
XVSS-X Pipeline Utilities: Manifest I/O, checkpoints, text normalization, and audio helpers.
"""

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import soundfile as sf
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("xvss-x")


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Normalize text for consistent sentence matching and alignment."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = text.strip('"\'')
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Load dataset manifest file (JSON)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(data: Dict[str, Any], path: Union[str, Path], indent: int = 2):
    """Save dataset manifest file (JSON) atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    tmp_path.replace(path)


def batch_iterator(items: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """Yield successive batches from items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load existing processing checkpoint."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read checkpoint {checkpoint_path}: {e}")
    return {"processed_ids": [], "results": []}


def save_checkpoint(checkpoint_path: Path, processed_ids: List[str], results: List[Dict[str, Any]]):
    """Save processing checkpoint atomically."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    data = {
        "processed_ids": processed_ids,
        "results": results,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp_path.replace(checkpoint_path)


def get_audio_info(audio_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Return sampling rate, duration, and channels for an audio file."""
    try:
        info = sf.info(str(audio_path))
        return {
            "duration": float(info.duration),
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
        }
    except Exception as e:
        logger.debug(f"Could not read audio info for {audio_path}: {e}")
        return None

