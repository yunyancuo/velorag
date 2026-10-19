"""Prepare the public BEIR SciFact training split with a provenance manifest.

The official test qrels remain in the archive and are not extracted by this tool.
This is a small research pilot dataset, not a large-corpus benchmark.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


SOURCE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
PUBLISHED_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
FILES = ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv")


def prepare(destination: Path):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "scifact.zip"
    if not archive_path.exists():
        temporary = destination / "scifact.zip.part"
        downloaded = 0
        with urlopen(SOURCE, timeout=30) as response, temporary.open("wb") as output:
            while block := response.read(256 * 1024):
                downloaded += len(block)
                if downloaded > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive exceeds the expected size bound")
                output.write(block)
        temporary.replace(archive_path)

    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive exceeds the expected size bound")
    raw = archive_path.read_bytes()
    if hashlib.md5(raw).hexdigest() != PUBLISHED_MD5:
        raise ValueError("Archive differs from the published BEIR checksum")

    extracted = {}
    with ZipFile(archive_path) as archive:
        for relative in FILES:
            member = archive.getinfo(f"scifact/{relative}")
            if member.file_size > 64 * 1024 * 1024:
                raise ValueError(f"Unexpected expanded size: {relative}")
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Destination leaves dataset directory")
            content = archive.read(member)
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite modified dataset file: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            extracted[relative] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "lines": len(content.splitlines()),
            }

    manifest = {
        "dataset": "BEIR SciFact",
        "source": SOURCE,
        "source_documentation": "https://github.com/beir-cellar/beir",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_md5": PUBLISHED_MD5,
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "split_extracted": "train",
        "test_qrels_extracted": False,
        "purpose": "small public research pilot; not evidence of large-corpus performance",
        "files": extracted,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/scifact"))
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.out), indent=2))
