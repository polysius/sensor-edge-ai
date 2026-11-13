"""Download NASA C-MAPSS, verify both archives against known hashes, and unpack the tables."""

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

# The NASA data portal link returns 403 for scripts, so this uses the PHM data mirror.
OUTER_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
OUTER_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_NAME = "CMAPSSData.zip"
INNER_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path.name}: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss"))
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    outer_zip = data_dir / "outer.zip"
    if not outer_zip.exists():
        print(f"Downloading {OUTER_URL}")
        urllib.request.urlretrieve(OUTER_URL, outer_zip)
    verify(outer_zip, OUTER_SHA256)

    with zipfile.ZipFile(outer_zip) as outer:
        inner_member = next(name for name in outer.namelist() if name.endswith(INNER_NAME))
        inner_zip = data_dir / INNER_NAME
        inner_zip.write_bytes(outer.read(inner_member))
    verify(inner_zip, INNER_SHA256)

    with zipfile.ZipFile(inner_zip) as inner:
        inner.extractall(data_dir)
    print(f"Verified and unpacked C-MAPSS into {data_dir}")


if __name__ == "__main__":
    main()
