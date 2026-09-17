"""저장 디렉터리 공통: sha256.txt 쓰기·검증."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

SHA256_NAME = "sha256.txt"


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256(d: Path, files: Iterable[str]) -> None:
    lines = [f"{sha256_file(Path(d) / name)}  {name}" for name in files]
    (Path(d) / SHA256_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_sha256(d: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (Path(d) / SHA256_NAME).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out


def sha256_problems(d: Path, files: Iterable[str]) -> list[str]:
    d = Path(d)
    if not (d / SHA256_NAME).exists():
        return [f"{SHA256_NAME} 없음"]
    recorded = read_sha256(d)
    p = []
    for name in files:
        if name not in recorded:
            p.append(f"sha256.txt 에 {name} 없음")
        elif not (d / name).exists():
            p.append(f"{name} 파일 없음")
        elif sha256_file(d / name) != recorded[name]:
            p.append(f"{name} sha256 불일치")
    return p


def verify_sha256(d: Path, files: Iterable[str]) -> None:
    p = sha256_problems(d, files)
    if p:
        raise ValueError("; ".join(p))
