"""Pinned public datasets; explicit mirror selection and verified content hashes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import zlib


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, destination: Path, size: int, expected: str, algorithm="sha256") -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != size or digest(destination, algorithm) != expected:
            raise ValueError(f"Existing dataset failed checksum: {destination}")
        return {"file": destination.name, "size": size, "checksum": expected, "status": "verified_cached"}
    partial = destination.with_suffix(destination.suffix + ".part")
    if not partial.exists() or partial.stat().st_size != size:
        command = ["curl", "--fail", "--location", "--connect-timeout", "30", "--max-time", "3600", "--output", str(partial)]
        if partial.exists() and partial.stat().st_size > 0:
            command += ["--continue-at", "-"]
        subprocess.run(command + [url], check=True)
    if partial.stat().st_size != size or digest(partial, algorithm) != expected:
        raise ValueError(f"Downloaded dataset failed checksum: {partial}; kept for inspection")
    partial.replace(destination)
    return {"file": destination.name, "size": size, "checksum": expected, "algorithm": algorithm, "status": "verified_download", "url": url}


def safe_extract(archive: Path, target: Path, max_bytes=8 * 1024**3) -> None:
    """Reject traversal, symlinks and oversized archives; never delete existing files."""
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        total = 0
        for member in zf.infolist():
            normalized = member.filename.replace("\\", "/")
            if normalized.startswith("/") or ":" in normalized or ".." in Path(normalized).parts:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            resolved = (root / normalized).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"ZIP traversal: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"ZIP symlink: {member.filename}")
            total += member.file_size
            if total > max_bytes:
                raise ValueError("Archive exceeds extraction budget")
        for member in zf.infolist():
            resolved = root / member.filename.replace("\\", "/")
            if member.is_dir():
                resolved.mkdir(parents=True, exist_ok=True)
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if resolved.exists():
                crc = 0
                with resolved.open("rb") as existing:
                    for block in iter(lambda: existing.read(1024**2), b""):
                        crc = zlib.crc32(block, crc)
                if resolved.stat().st_size != member.file_size or (crc & 0xffffffff) != member.CRC:
                    raise ValueError(f"Refusing to overwrite conflicting file: {resolved}")
                continue
            with zf.open(member) as source, resolved.open("xb") as output:
                shutil.copyfileobj(source, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["cord", "fatura", "all"], default="all")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--endpoint", choices=["https://huggingface.co", "https://hf-mirror.com"], default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    sources = json.loads((root / "configs/data_sources.json").read_text(encoding="utf-8"))
    result = {"endpoint": args.endpoint, "files": [], "sources": []}
    if args.source in ("cord", "all"):
        source = sources["cord"]
        folder = root / "data/raw/cord"
        def download_one(item):
            url = f"{args.endpoint}/datasets/{source['source']}/resolve/{source['revision']}/{item['path']}"
            return fetch(url, folder / Path(item["path"]).name, item["size"], item["sha256"])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            result["files"].extend(pool.map(download_one, source["files"]))
        result["sources"].append("cord")
    if args.source in ("fatura", "all"):
        source = sources["fatura"]
        archive = root / "data/raw" / source["filename"]
        result["files"].append(fetch(source["url"], archive, source["size"], source["md5"], "md5"))
        result["sources"].append("fatura")
        if args.extract:
            target = root / "data/raw/fatura"
            safe_extract(archive, target)
            # FATURA distributions contain explicitly named nested ZIPs.
            nested = sorted(target.glob("*.zip"))
            for item in nested:
                safe_extract(item, target)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / f"download_{args.source}.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
