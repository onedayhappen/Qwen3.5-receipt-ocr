import io
import json
from pathlib import Path

import pytest
from PIL import Image

from receipt.inference import OpenAIReceiptClient, prepare_image, run_batch
from receipt.train_launcher import build_command
from receipt.download import safe_extract


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (400, 800), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_pixel_grid_and_input_safety():
    image, metadata = prepare_image(image_bytes(), 602112)
    assert image.width % 32 == image.height % 32 == 0
    assert metadata["pixels"] <= 602112
    with pytest.raises(ValueError):
        prepare_image(image_bytes(), 602113)
    with pytest.raises(ValueError):
        prepare_image(b"not image")


def test_launcher_repo_relative_and_no_test(tmp_path):
    data = tmp_path / "train.json"
    data.write_text('[{"split":"train"}]')
    config = {"model_id": "Qwen/Qwen3.5-4B", "disable_flash_attn2": True, "data_path": "train.json", "output_dir": "output/sft"}
    command, options = build_command(config, tmp_path, "sft")
    assert options["data_path"] == str(data.resolve())
    assert command[1:3] == ["-m", "train.train_sft"]
    data.write_text('[{"split":"test"}]')
    with pytest.raises(ValueError, match="Test"):
        build_command(config, tmp_path, "sft")


def test_grpo_generation_divisibility(tmp_path):
    config = {"model_id": "models/sft-merged", "disable_flash_attn2": True, "num_generations": 4,
              "per_device_train_batch_size": 1, "gradient_accumulation_steps": 3}
    with pytest.raises(ValueError, match="divisible"):
        build_command(config, tmp_path, "grpo")


def test_zip_traversal_rejected(tmp_path):
    import zipfile
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "escape.txt").exists()


def test_zip_existing_same_size_corruption_is_rejected(tmp_path):
    import zipfile
    archive = tmp_path / "normal.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("data.txt", "gold")
    target = tmp_path / "extract"
    target.mkdir()
    (target / "data.txt").write_text("evil")
    with pytest.raises(ValueError, match="conflicting"):
        safe_extract(archive, target)
    assert (target / "data.txt").read_text() == "evil"


class FakeBackend:
    def __init__(self, response='{"total":"25.00"}', finish_reason="stop", ready=True):
        self.response, self.finish_reason, self.ready = response, finish_reason, ready
    def health(self):
        return self.ready
    def extract(self, data, schema_version="receipt_v1", fields=None):
        prepare_image(data)
        return {"response": self.response, "finish_reason": self.finish_reason, "latency_ms": 1,
                "model_version": "fake-not-a-model"}


def test_api_valid_partial_requires_review():
    from receipt.service import create_app
    client = create_app(FakeBackend()).test_client()
    response = client.post("/extract", data={"image": (io.BytesIO(image_bytes()), "receipt.png"), "fields": "total"})
    assert response.status_code == 200
    assert response.json["status"] == "review_required"
    assert response.json["validation"]["schema_valid"]
    assert "uncalibrated_requires_review" in response.json["validation"]["issues"]


def test_api_invalid_json_and_truncation_fail():
    from receipt.service import create_app
    for backend in (FakeBackend("```json\n{}\n```"), FakeBackend(finish_reason="length")):
        response = create_app(backend).test_client().post("/extract", data={"image": (io.BytesIO(image_bytes()), "receipt.png")})
        assert response.status_code == 422
        assert response.json["status"] == "failed"


def test_api_schema_invalid_is_not_success():
    from receipt.service import create_app
    response = create_app(FakeBackend('{"total":25}')).test_client().post(
        "/extract", data={"image": (io.BytesIO(image_bytes()), "receipt.png"), "fields": "total"})
    assert response.status_code == 422
    assert response.json["status"] == "failed"
    assert not response.json["validation"]["schema_valid"]


def test_shared_training_inference_pixel_limits():
    from receipt.inference import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS
    repo = Path(__file__).resolve().parents[2]
    for filename in ("sft.json", "dpo.json", "grpo.json", "vision_lora.json"):
        config = json.loads((repo / "configs/receipt" / filename).read_text())
        assert config["image_min_pixels"] == DEFAULT_MIN_PIXELS
        assert config["image_max_pixels"] == DEFAULT_MAX_PIXELS
    image, metadata = prepare_image(image_bytes(), DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS)
    assert metadata["resize_policy"] == "qwen_smart_resize_32"
    assert DEFAULT_MIN_PIXELS <= image.width * image.height <= DEFAULT_MAX_PIXELS


def test_online_and_local_byte_limits_are_separate():
    from receipt.inference import MAX_UPLOAD_BYTES, MAX_LOCAL_IMAGE_BYTES
    payload = image_bytes() + b"\0" * MAX_UPLOAD_BYTES
    with pytest.raises(ValueError, match="input limit"):
        prepare_image(payload)
    image, _ = prepare_image(payload, max_bytes=MAX_LOCAL_IMAGE_BYTES)
    assert image.width > 0
    from receipt.service import create_app
    response = create_app(FakeBackend()).test_client().post(
        "/extract", data={"image": (io.BytesIO(payload), "large.png"), "fields": "total"})
    assert response.status_code == 413


def test_shared_local_training_preprocessor_without_gpu():
    import ast
    from receipt.inference import DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
    repo = Path(__file__).resolve().parents[2]
    tree = ast.parse((repo / "src/dataset/data_utils.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_image_info")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "training_image_preprocessor", "exec"), namespace)
    # PIL bytes only; no torch or qwen_vl_utils required for local mainline.
    import uuid
    path = repo / ".pytest_runs" / ("preprocessor-" + uuid.uuid4().hex + ".png")
    path.write_bytes(image_bytes())
    trained = namespace["get_image_info"](str(path), DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS, None, None, 16)
    inferred, _ = prepare_image(path.read_bytes(), DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS)
    assert trained.size == inferred.size and trained.tobytes() == inferred.tobytes()


def test_health_reports_backend_failure():
    from receipt.service import create_app
    response = create_app(FakeBackend(ready=False)).test_client().get("/health")
    assert response.status_code == 503
    assert not response.json["backend_ready"]


def test_batch_never_sends_annotation_and_test_guard(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images/a.png").write_bytes(image_bytes())
    sample = {"sample_id": "a", "split": "test", "schema_version": "receipt_v1", "image": "images/a.png",
              "ground_truth": {"total": "25.00"}, "annotation": {"secret": "never_in_prompt"}, "requested_fields": ["total"]}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(sample) + "\n")
    with pytest.raises(ValueError, match="--allow-test"):
        run_batch(manifest, tmp_path / "out.jsonl", tmp_path, FakeBackend())
    assert run_batch(manifest, tmp_path / "out.jsonl", tmp_path, FakeBackend(), allow_test=True) == 1
    prediction = json.loads((tmp_path / "out.jsonl").read_text())
    assert "ground_truth" not in prediction
    assert "annotation" not in prediction


def test_openai_payload_is_nonthinking_and_image_only():
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": '{"total":"25.00"}'}, "finish_reason": "stop"}]}
    class Session:
        def post(self, url, json=None, timeout=None):
            self.payload = json
            assert timeout == 120
            return Response()
    session = Session()
    client = OpenAIReceiptClient(session=session)
    client.extract(image_bytes(), fields=["total"])
    assert session.payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert session.payload["messages"][0]["role"] == "user"
    assert "ground_truth" not in json.dumps(session.payload)
