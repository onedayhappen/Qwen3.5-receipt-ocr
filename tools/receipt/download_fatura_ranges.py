"""Download immutable Zenodo FATURA using bounded range chunks after dropped streams."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import subprocess
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from receipt.download import digest, safe_extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--url", help="Explicit alternative official URL; final official MD5 remains mandatory")
    args = parser.parse_args()
    source = json.loads((args.root / "configs/data_sources.json").read_text(encoding="utf-8"))["fatura"]
    url = args.url or source["url"]
    directory = args.root / "data/raw/fatura-chunks"
    directory.mkdir(parents=True, exist_ok=True)
    chunk_size = 4 * 1024**2
    chunks = [(i, start, min(source["size"] - 1, start + chunk_size - 1)) for i, start in enumerate(range(0, source["size"], chunk_size))]
    def fetch_chunk(chunk):
        index, start, end = chunk
        path = directory / f"{index:04d}.part"
        expected_size = end - start + 1
        if path.exists() and path.stat().st_size == expected_size:
            return path
        command = ["curl", "--fail", "--silent", "--show-error", "--location", "--connect-timeout", "30", "--max-time", "180",
                   "--range", f"{start}-{end}", "--output", str(path), url]
        result = subprocess.run(command, check=False)
        if result.returncode or not path.exists() or path.stat().st_size != expected_size:
            raise RuntimeError(f"Range failed {start}-{end}; cached completed chunks retained")
        return path
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(pool.map(fetch_chunk, chunks))
    archive = args.root / "data/raw/FATURA2.zip"
    if archive.exists():
        if digest(archive, "md5") != source["md5"]:
            raise ValueError("Existing archive checksum mismatch")
    else:
        assembled = archive.with_suffix(".zip.assembled.part")
        if not assembled.exists():
            with assembled.open("xb") as target:
                for path in paths:
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024**2), b""):
                            target.write(block)
        if assembled.stat().st_size != source["size"] or digest(assembled, "md5") != source["md5"]:
            raise ValueError("Assembled archive failed size/official MD5; retained partial for inspection")
        assembled.replace(archive)
    if digest(archive, "md5") != source["md5"]:
        raise ValueError("Assembled archive failed official MD5")
    extracted = args.root / "data/raw/fatura"
    safe_extract(archive, extracted)
    for path in sorted(extracted.glob("*.zip")):
        safe_extract(path, extracted)
    report = {"source":"FATURA", "size":source["size"], "md5":source["md5"], "status":"verified_download", "method":"bounded_http_ranges", "url":url}
    (args.root / "reports/download_fatura.json").write_text(json.dumps(report,indent=2) + "\n",encoding="utf-8")
    print(json.dumps(report))

if __name__ == "__main__":
    main()
