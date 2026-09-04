"""Verify the benchmark files against the S2RMS snapshot used in the paper."""

from __future__ import annotations

import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = REPOSITORY_ROOT / "data" / "SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures = 0
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            print(f"MISSING  {relative_path}")
            failures += 1
            continue

        actual = sha256(path)
        if actual == expected:
            print(f"OK       {relative_path}")
        else:
            print(f"MISMATCH {relative_path}")
            print(f"         expected: {expected}")
            print(f"         actual:   {actual}")
            failures += 1

    if failures:
        print(f"\nVerification failed for {failures} file(s).")
        return 1

    print("\nAll benchmark files match the reference S2RMS snapshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
