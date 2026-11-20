"""Standalone local inference adapters; no training labels sent to a model backend."""
from __future__ import annotations
import argparse
import base64
import io
import json
import math
from pathlib import Path
import time

import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from receipt.prompts import build_prompt
from receipt.schema import get_schema

MAX_UPLOAD_BYTES = 8 * 1024**2
MAX_LOCAL_IMAGE_BYTES = 32 * 1024**2
MAX_DECODED_PIXELS = 40_000_000
DEFAULT_MAX_PIXELS = 1048576
DEFAULT_MIN_PIXELS = 65536


def prepare_image(image_bytes: bytes, max_pixels=602112, min_pixels=None, max_bytes=MAX_UPLOAD_BYTES) -> tuple[Image.Image, dict]:
    if not image_bytes or len(image_bytes) > max_bytes:
        raise ValueError(f"Image is empty or exceeds the {max_bytes} byte input limit")
    if max_pixels < 4096 or max_pixels % 1024:
        raise ValueError("Qwen3.5 max_pixels must be >=4096 and aligned to 1024")
    try:
        with Image.open(io.BytesIO(image_bytes)) as original:
            if original.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Only JPEG, PNG and WEBP images are accepted")
            width, height = original.size
            if width * height > MAX_DECODED_PIXELS or min(width, height) < 1:
                raise ValueError("Decoded image exceeds safety limit")
            image = ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or unsafe image") from exc
    old_size = image.size
    if min_pixels is not None:
        if min_pixels < 1024 or min_pixels % 1024 or min_pixels > max_pixels:
            raise ValueError("Invalid aligned minimum pixel budget")
        # Same smart_resize contract as training's qwen_vl_utils, without importing torch.
        if max(image.width, image.height) / min(image.width, image.height) > 200:
            raise ValueError("Image aspect ratio exceeds training processor limit")
        new_w = max(32, round(image.width / 32) * 32)
        new_h = max(32, round(image.height / 32) * 32)
        if new_w * new_h > max_pixels:
            beta = math.sqrt(image.width * image.height / max_pixels)
            new_w = max(32, math.floor(image.width / beta / 32) * 32)
            new_h = max(32, math.floor(image.height / beta / 32) * 32)
        elif new_w * new_h < min_pixels:
            beta = math.sqrt(min_pixels / (image.width * image.height))
            new_w = math.ceil(image.width * beta / 32) * 32
            new_h = math.ceil(image.height * beta / 32) * 32
        if new_w * new_h > max_pixels:
            raise ValueError("Extreme aspect ratio cannot fit aligned training pixel budget")
        if image.size != (new_w, new_h):
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return image, {"original_size": list(old_size), "processed_size": list(image.size),
                       "pixels": image.width * image.height, "vision_tokens": image.width * image.height // 1024,
                       "resize_policy": "qwen_smart_resize_32", "min_pixels": min_pixels, "max_pixels": max_pixels}
    # Compatibility for callers that explicitly want shrink-only preparation.
    # Model adapters always provide the shared training minimum and maximum.
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    new_w = max(32, int(image.width * scale / 32) * 32)
    new_h = max(32, int(image.height * scale / 32) * 32)
    while new_w * new_h > max_pixels:
        if new_w >= new_h and new_w > 32:
            new_w -= 32
        elif new_h > 32:
            new_h -= 32
        else:
            raise ValueError("Cannot fit image in pixel budget")
    if image.size != (new_w, new_h):
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return image, {"original_size": list(old_size), "processed_size": list(image.size),
                   "pixels": image.width * image.height, "vision_tokens": image.width * image.height // 1024}


class OpenAIReceiptClient:
    def __init__(self, base_url="http://127.0.0.1:8080/v1", model="document-extraction", timeout=120,
                 max_pixels=DEFAULT_MAX_PIXELS, max_tokens=2048, constrained=False, session=None, min_pixels=DEFAULT_MIN_PIXELS):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_tokens = max_tokens
        self.constrained = constrained
        self.session = session or requests.Session()
        self.session.trust_env = False  # local inference should not be sent to a configured HTTP proxy

    def health(self):
        try:
            response = self.session.get(self.base_url + "/models", timeout=min(5, self.timeout))
            return response.status_code == 200
        except requests.RequestException:
            return False

    def extract(self, image_bytes, schema_version="receipt_v1", fields=None):
        started = time.perf_counter()
        image, metadata = prepare_image(image_bytes, self.max_pixels, self.min_pixels, max_bytes=MAX_LOCAL_IMAGE_BYTES)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}},
                   {"type": "text", "text": build_prompt(schema_version=schema_version, fields=fields)}]
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0.0,
                   "max_tokens": self.max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        if self.constrained:
            payload["structured_outputs"] = {"json": get_schema(schema_version=schema_version, fields=fields)}
        response = self.session.post(self.base_url + "/chat/completions", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        raw = choice["message"].get("content") or ""
        return {"response": raw, "finish_reason": choice.get("finish_reason"), "model_version": self.model,
                "latency_ms": (time.perf_counter() - started) * 1000, "usage": data.get("usage", {}),
                "image": metadata, "constrained_decoding": self.constrained}


class TransformersReceiptClient:
    """Lazy GPU dependency loading; safe CPU tooling imports without torch."""
    def __init__(self, model, max_pixels=DEFAULT_MAX_PIXELS, max_tokens=2048, min_pixels=DEFAULT_MIN_PIXELS):
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        config = AutoConfig.from_pretrained(model)
        if config.model_type not in {"qwen3_5", "qwen3_5_moe"}:
            raise ValueError("Only Qwen3.5 dense/MoE supported")
        self.model_name = model
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_tokens = max_tokens

    def health(self):
        return True

    def extract(self, image_bytes, schema_version="receipt_v1", fields=None):
        import torch
        started = time.perf_counter()
        image, metadata = prepare_image(image_bytes, self.max_pixels, self.min_pixels, max_bytes=MAX_LOCAL_IMAGE_BYTES)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                    {"type": "text", "text": build_prompt(schema_version=schema_version, fields=fields)}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[prompt], images=[image], do_resize=False, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        completion = output[0, inputs["input_ids"].shape[-1]:]
        raw = self.processor.decode(completion, skip_special_tokens=True)
        finish = "length" if completion.numel() >= self.max_tokens else "stop"
        return {"response": raw, "finish_reason": finish, "model_version": self.model_name,
                "latency_ms": (time.perf_counter() - started) * 1000, "image": metadata,
                "usage": {"completion_tokens": completion.numel()}, "constrained_decoding": False}


def run_batch(manifest, output, image_root, client, allow_test=False):
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not allow_test and any(row.get("split") == "test" for row in rows):
        raise ValueError("Locked test inference requires explicit --allow-test")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs in batch manifest")
    root = Path(image_root).resolve()
    from receipt.data import initialize_source_schemas
    initialize_source_schemas(rows, root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Prediction file exists; choose a new run path: {output}")
    with output.open("x", encoding="utf-8") as handle:
        for sample in rows:
            image = (root / sample["image"]).resolve()
            if not image.is_relative_to(root):
                raise ValueError("Manifest image escapes image_root")
            try:
                result = client.extract(image.read_bytes(), sample.get("schema_version", "receipt_v1"), sample.get("requested_fields"))
            except (requests.RequestException, ValueError, OSError, KeyError, IndexError, TypeError) as exc:
                result = {"response": "", "finish_reason": "error", "error_type": type(exc).__name__, "latency_ms": None}
            from receipt.data import render_user, sha256_bytes
            prompt = render_user(sample)
            result.update({"sample_id": sample["sample_id"], "split": sample["split"], "schema_version": sample["schema_version"],
                           "requested_fields": sample.get("requested_fields"), "prompt": prompt,
                           "prompt_version": "sha256:" + sha256_bytes(prompt.encode("utf-8")), "reviewed": False})
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["openai", "transformers"], default="openai")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="document-extraction")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(argv)
    client = OpenAIReceiptClient(args.base_url, args.model, args.timeout, args.max_pixels, args.max_tokens, args.constrained, min_pixels=args.min_pixels) if args.backend == "openai" else TransformersReceiptClient(args.model, args.max_pixels, args.max_tokens, args.min_pixels)
    print(json.dumps({"predictions": run_batch(args.manifest, args.output, args.image_root, client, args.allow_test), "output": str(args.output)}))

if __name__ == "__main__":
    main()
