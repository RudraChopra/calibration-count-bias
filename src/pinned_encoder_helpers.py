"""Frozen, resumable extraction for the final VocabShift GPU study.

This module never substitutes CPU measurements for GPU results. Its numerical
and checkpoint helpers can be tested without importing torch or transformers.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time
import uuid
from typing import Any, Callable

import numpy as np

MODELS = [
    {"id": "openai/clip-vit-base-patch32", "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "kind": "clip", "safetensors": False},
    {"id": "openai/clip-vit-large-patch14", "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41", "kind": "clip", "safetensors": True},
    {"id": "google/siglip-base-patch16-224", "revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", "kind": "siglip", "safetensors": True},
]
EXPECTED_DATASETS = {"OxfordPets", "EuroSAT"}
PROMPT = "a photo of a {}."
EVIDENCE_ORIGIN = "NEW_FINAL_STUDY_CUDA_EXTRACTION"
CHECKPOINT_IMAGES = 256
EXPECTED_WEIGHT_SHA256 = {
    "openai/clip-vit-base-patch32": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
    "openai/clip-vit-large-patch14": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
    "google/siglip-base-patch16-224": "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def receipt_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256.json")


def write_verified_npz(path: Path, identity: str, **arrays: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    record = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "identity_sha256": identity}
    atomic_json(receipt_path(path), record)
    return record


def read_verified_npz(path: Path, identity: str) -> dict[str, np.ndarray] | None:
    path = Path(path)
    receipt = receipt_path(path)
    if not path.exists():
        if receipt.exists():
            raise RuntimeError(f"Receipt exists but its committed artifact is missing: {path}")
        return None
    if not receipt.exists():
        # A process can die between the two atomic commits. Uncommitted bytes
        # are never trusted or blessed with a newly manufactured receipt.
        quarantine = path.with_name(path.name + ".uncommitted." + uuid.uuid4().hex)
        os.replace(path, quarantine)
        print(f"Quarantined incomplete commit and will recompute: {path.name}", flush=True)
        return None
    record = json.loads(receipt.read_text())
    if record.get("identity_sha256") != identity:
        raise RuntimeError(f"Frozen identity mismatch: {path.name}")
    if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
        raise RuntimeError(f"SHA256 or byte count mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key].copy() for key in a.files}


def normalized_features(array: np.ndarray, name: str) -> None:
    if array.dtype != np.float32 or array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Invalid finite float32 feature matrix: {name}")
    if not np.allclose(np.linalg.norm(array.astype(np.float64), axis=1), 1.0, atol=2e-4, rtol=0):
        raise ValueError(f"Features are not normalized: {name}")


def compute_logits(image: np.ndarray, text: np.ndarray, scale: float, bias: float) -> np.ndarray:
    normalized_features(image, "images")
    normalized_features(text, "text")
    if image.shape[1] != text.shape[1] or not np.isfinite([scale, bias]).all() or scale <= 0:
        raise ValueError("Invalid feature dimensions or learned logit scale/bias")
    logits = (image.astype(np.float64) @ text.astype(np.float64).T * scale + bias).astype(np.float32)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite canonical logits")
    return logits


def validate_cache(cache: dict[str, np.ndarray], dataset: dict[str, Any], identity: str, model: str) -> None:
    required = {"image_features", "text_features", "logits", "image_ids", "labels", "roles", "scale", "bias", "metadata"}
    if not required.issubset(cache):
        raise ValueError(f"Missing cache keys: {sorted(required - set(cache))}")
    meta = json.loads(str(cache["metadata"].item()))
    if meta.get("identity_sha256") != identity or meta.get("model") != model or meta.get("dataset") != dataset["id"]:
        raise ValueError("Cache metadata does not match its frozen model/dataset identity")
    if meta.get("evidence_origin") != EVIDENCE_ORIGIN or meta.get("device") != "cuda:0":
        raise ValueError("Cache is not a recorded CUDA extraction")
    rows = dataset["rows"]
    for key, expected in [("image_ids", [r["id"] for r in rows]), ("labels", [r["label"] for r in rows]), ("roles", [r["role"] for r in rows])]:
        if not np.array_equal(cache[key], np.asarray(expected)):
            raise ValueError(f"Cache row alignment mismatch: {key}")
    if cache["logits"].shape != (len(rows), len(dataset["class_names"])):
        raise ValueError("Logit shape differs from manifest")
    if cache["image_features"].shape[0] != len(rows) or cache["text_features"].shape[0] != len(dataset["class_names"]):
        raise ValueError("Feature rows differ from manifest")
    expected = compute_logits(cache["image_features"], cache["text_features"], float(cache["scale"].item()), float(cache["bias"].item()))
    if cache["logits"].dtype != np.float32 or not np.array_equal(cache["logits"], expected):
        raise ValueError("Canonical logits do not reconstruct exactly from stored features")


def guarded_image_path(data_root: Path, relative: str) -> Path:
    path = (data_root / relative).resolve()
    if not path.is_relative_to(data_root.resolve()):
        raise ValueError("Image manifest path escapes the declared data root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_images(data_root: Path, datasets: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    records = {}
    for name, dataset in datasets.items():
        rows = dataset["rows"]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate image IDs: {name}")
        selected = []
        for i, row in enumerate(rows):
            path = guarded_image_path(data_root, row["path"])
            actual = sha256_file(path)
            if actual != row["sha256"]:
                raise ValueError(f"Image SHA256 mismatch: {name}/{row['id']}")
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                rgb.load()
                size = list(rgb.size)
                rgb.close()
            if "rgb_size" in row and size != row["rgb_size"]:
                raise ValueError(f"Decoded dimensions changed: {name}/{row['id']}")
            if row["role"] not in {"cal_pool", "test"} or not 0 <= row["label"] < len(dataset["class_names"]):
                raise ValueError("Invalid image role or label")
            selected.append({"id": row["id"], "sha256": actual, "rgb_size": size})
            if (i + 1) % 1000 == 0:
                print(f"Verified image bytes and RGB decode: {name} {i + 1}/{len(rows)}", flush=True)
        records[name] = selected
    return {"status": "COMPLETE", "datasets": records, "selected_images": sum(map(len, records.values()))}


def retry_download(function: Callable[[], Any], attempts: int = 4) -> Any:
    for attempt in range(attempts):
        try:
            return function()
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            name = type(error).__name__
            transient = status == 429 or (isinstance(status, int) and status >= 500) or name in {
                "ConnectTimeout", "ReadTimeout", "Timeout", "ConnectError", "ConnectionError", "RemoteProtocolError", "ReadError",
            }
            if not transient or attempt + 1 == attempts:
                raise
            delay = min(15, 2 ** attempt)
            print(f"Transient checkpoint download failure ({name}); retrying in {delay}s", flush=True)
            time.sleep(delay)


def prepare_model_assets(hf_cache: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    assets = {}
    for spec in MODELS:
        weight = "model.safetensors" if spec["safetensors"] else "pytorch_model.bin"
        patterns = [weight, "config.json", "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "spiece.model", "sentencepiece.bpe.model", "added_tokens.json"]
        print(f"Fetching pinned public model assets: {spec['id']}", flush=True)
        location = retry_download(lambda: snapshot_download(repo_id=spec["id"], revision=spec["revision"], cache_dir=str(hf_cache), allow_patterns=patterns, token=False, max_workers=4))
        local = Path(location)
        if not (local / weight).is_file() or not (local / "config.json").is_file():
            raise FileNotFoundError(f"Pinned model snapshot is incomplete: {spec['id']}")
        files = {str(p.relative_to(local)): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(local.rglob("*")) if p.is_file()}
        if files[weight]["sha256"] != EXPECTED_WEIGHT_SHA256[spec["id"]]:
            raise RuntimeError(f"Pinned weight SHA256 mismatch: {spec['id']}")
        assets[spec["id"]] = {"revision": spec["revision"], "local_path": str(local), "files": files, "safetensors": spec["safetensors"]}
    return assets


def configure_torch() -> tuple[Any, dict[str, Any]]:
    import torch
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < (2, 6):
        raise RuntimeError("The pinned CLIP B32 binary requires torch >= 2.6. Use the supplied isolated CUDA environment.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU extraction requires CUDA. CPU software tests cannot substitute for this experiment.")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(74019)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    packages = {}
    for name in ["torch", "torchvision", "transformers", "numpy", "Pillow", "huggingface-hub", "safetensors", "sentencepiece", "protobuf", "tokenizers"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    if packages["transformers"] != "4.57.1":
        raise RuntimeError("Use transformers==4.57.1 from the supplied notebook setup.")
    device = torch.cuda.get_device_properties(0)
    environment = {"python": platform.python_version(), "packages": packages, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": "cuda:0", "gpu_name": device.name, "gpu_capability": list(torch.cuda.get_device_capability(0)), "gpu_memory_bytes": device.total_memory, "tf32": False, "forward_dtype": "float32", "attention": "eager", "logit_arithmetic": "float64 dot product and scale/bias, stored float32"}
    return torch, environment


def import_models() -> tuple[Any, Any, Any]:
    import transformers.utils.import_utils as iu
    # These encoders do not use optional audio backends, which can be broken in
    # otherwise valid notebook images. This changes dependency discovery only.
    iu._librosa_available = False
    if hasattr(iu, "_torchcodec_available"):
        iu._torchcodec_available = False
    from transformers import AutoProcessor, CLIPModel, SiglipModel
    return AutoProcessor, CLIPModel, SiglipModel


def encode_text(model: Any, processor: Any, spec: dict[str, Any], class_names: list[str], torch: Any) -> np.ndarray:
    import torch.nn.functional as F
    values = []
    prompts = [PROMPT.format(x) for x in class_names]
    for start in range(0, len(prompts), 32):
        batch = prompts[start:start + 32]
        encoded = processor(text=batch, return_tensors="pt", padding="max_length" if spec["kind"] == "siglip" else True, max_length=64 if spec["kind"] == "siglip" else 77, truncation=True)
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        with torch.inference_mode():
            features = model.get_text_features(**encoded)
            if not torch.is_tensor(features):
                raise RuntimeError("Pinned text API did not return a tensor")
            values.append(F.normalize(features.float(), dim=-1).cpu().numpy().astype(np.float32))
    result = np.concatenate(values)
    normalized_features(result, "text")
    return result


def encode_images(model: Any, processor: Any, rows: list[dict[str, Any]], data_root: Path, torch: Any, batch_size: int, batch_events: list[dict[str, Any]], dataset_id: str, offset: int) -> tuple[np.ndarray, int]:
    from PIL import Image
    import torch.nn.functional as F
    values = []
    cursor = 0
    while cursor < len(rows):
        stop = min(len(rows), cursor + batch_size)
        images = []
        encoded = pixels = features = None
        try:
            for row in rows[cursor:stop]:
                with Image.open(guarded_image_path(data_root, row["path"])) as source:
                    rgb = source.convert("RGB")
                    rgb.load()
                    images.append(rgb)
            encoded = processor(images=images, return_tensors="pt")
            pixels = encoded["pixel_values"].to("cuda:0", dtype=torch.float32)
            with torch.inference_mode():
                features = model.get_image_features(pixel_values=pixels)
                if not torch.is_tensor(features):
                    raise RuntimeError("Pinned image API did not return a tensor")
                array = F.normalize(features.float(), dim=-1).cpu().numpy().astype(np.float32)
            normalized_features(array, "image batch")
            values.append(array)
            cursor = stop
        except torch.cuda.OutOfMemoryError:
            encoded = pixels = features = None
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            previous = batch_size
            batch_size = max(1, batch_size // 2)
            batch_events.append({"dataset": dataset_id, "row_offset": offset + cursor, "previous": previous, "new": batch_size, "reason": "CUDA_OutOfMemoryError"})
            print(f"CUDA memory retry with batch size {batch_size}; all images retained", flush=True)
        finally:
            encoded = pixels = features = None
            for image in images:
                image.close()
    result = np.concatenate(values)
    if len(result) != len(rows):
        raise AssertionError("Image batching omitted or duplicated rows")
    return result, batch_size


