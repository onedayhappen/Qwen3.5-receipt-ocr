"""CPU acceptance evidence; no GPU/model loading and no destructive cleanup."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from receipt.data import read_jsonl, write_json, audit_rows
from receipt.inference import prepare_image, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS, MAX_LOCAL_IMAGE_BYTES
from receipt.train_launcher import build_command


def run(root, wheel_target):
    root = root.resolve()
    tests = subprocess.run([sys.executable, "-m", "pytest", str(root / "tests"), "-q"], cwd=root, capture_output=True, text=True)
    compile_result = subprocess.run([sys.executable, "-m", "compileall", "-q", str(root / "src"), str(root / "tools"), str(root / "tests")], capture_output=True, text=True)
    rows = read_jsonl(root / "data/receipt/manifests/manifest.jsonl")
    audit = audit_rows(rows, root / "data/receipt")
    configs = {}
    for path in sorted((root / "configs/receipt").glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        stage = "sft" if path.stem == "vision_lora" else "dpo" if path.stem.startswith("dpo") else path.stem
        command, _ = build_command(config, root, stage)
        configs[path.name] = {"stage": stage, "launch_prefix": command[:3], "status": "dry_run_passed_not_gpu_verified"}
    preprocessed, failures = 0, []
    for path in sorted((root / "data/receipt/images").glob("*")):
        try:
            _, metadata = prepare_image(path.read_bytes(), DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, max_bytes=MAX_LOCAL_IMAGE_BYTES)
            assert metadata["pixels"] <= DEFAULT_MAX_PIXELS
            preprocessed += 1
        except (ValueError, OSError, AssertionError) as exc:
            failures.append({"image": path.name, "error": str(exc)})
    # Runtime-source scan is restricted to this project, not users' personal directories.
    runtime_files = [p for base in ("src", "tools", "scripts", "configs") for p in (root / base).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".sh", ".json", ".toml"}]
    external_dependencies = []
    for path in runtime_files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"Qwen3\.5-VL-Finetune|D:[/\\]GitHub[/\\]qwen-receipt-ocr", line):
                external_dependencies.append({"file": path.relative_to(root).as_posix(), "line": number, "text": line})
    isolation_code = (
        "import sys,json,pathlib; target=pathlib.Path(sys.argv[1]); sys.path.insert(0,str(target)); "
        "import receipt,receipt.schema,receipt.rewards,receipt.inference,receipt.train_launcher; "
        "from receipt.service import create_app; "
        "assert pathlib.Path(receipt.__file__).is_relative_to(target); assert 'torch' not in sys.modules; "
        "assert receipt.schema.validate_output({'total':'0'},fields=['total'])==[]; "
        "print(json.dumps({'module_path':receipt.__file__,'torch_imported':False,'isolated':sys.flags.isolated}))"
    )
    isolated = subprocess.run([sys.executable, "-I", "-c", isolation_code, str(wheel_target.resolve())], cwd=root, capture_output=True, text=True)
    wheels = sorted((root / "dist").glob("qwen35_receipt_ocr-*.whl"))
    wheel = wheels[-1]
    with zipfile.ZipFile(wheel) as archive:
        packaged = archive.namelist()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    cord = json.loads((root / "data/receipt/manifests/cord-report.json").read_text(encoding="utf-8"))
    status = tests.returncode == compile_result.returncode == isolated.returncode == 0 and not audit["errors"] and not failures and not external_dependencies
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "cpu_code_and_cord_accepted_with_explicit_limitations" if status else "failed",
        "python": sys.version, "python_executable": sys.executable,
        "system_packages_inherited": "include-system-site-packages = true" in (Path(sys.prefix) / "pyvenv.cfg").read_text(),
        "tests": {"exit_code": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr},
        "compile": {"exit_code": compile_result.returncode, "stderr": compile_result.stderr},
        "data_audit": audit, "raw_images_preprocessed": preprocessed, "preprocessing_failures": failures,
        "cord": {k: cord[k] for k in ("raw_source_samples", "samples", "split_counts", "suppressed_item_documents", "supervised_item_documents", "excluded")},
        "configs": configs, "source_runtime_dependencies": external_dependencies,
        "wheel": {"file": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "members": len(packaged),
                  "licenses": [n for n in packaged if "/licenses/" in n]},
        "isolated_wheel": {"exit_code": isolated.returncode, "stdout": isolated.stdout, "stderr": isolated.stderr},
        "installed_dependencies": freeze.stdout.splitlines(),
        "gpu_training_run": False, "real_model_inference_run": False, "merge_numeric_equivalence_verified": False,
        "fatura_verified_download": False, "model_generated_reviewed_preferences": False,
        "limitations": ["FATURA download incomplete; not in train data", "No Linux/CUDA training dependency import or GPU smoke",
                        "No model accuracy result", "CORD public labels do not supervise merchant/date/address",
                        "Official CORD splits do not guarantee merchant OOD", "210 item tasks suppressed to avoid unknown null supervision",
                        "8 raw records excluded with source GT retained; 98-sample test is a coverage subset, not official full benchmark",
                        "Synthetic preference errors are not reviewed model generations", "No review UI, persistent feedback or production authentication"]
    }
    write_json(root / "reports/acceptance.json", report)
    print(json.dumps({"status": report["status"], "tests": tests.stdout.strip().splitlines()[-1], "raw_images_preprocessed": preprocessed,
                      "audit_errors": len(audit["errors"]), "source_runtime_dependencies": external_dependencies,
                      "isolated_wheel_exit_code": isolated.returncode}, ensure_ascii=False, indent=2))
    if not status:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--wheel-target", type=Path, required=True, help="Fresh directory populated by pip install --target of final wheel")
    args = parser.parse_args()
    run(args.root, args.wheel_target)


if __name__ == "__main__":
    main()
