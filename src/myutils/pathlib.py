from collections.abc import Collection
from pathlib import Path


def rglob_with_suffices(path: Path, suffices: Collection[str]):
    return (p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffices)
