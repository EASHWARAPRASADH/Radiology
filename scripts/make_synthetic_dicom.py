"""Generate synthetic, PHI-free chest X-ray DICOMs for ingestion/API tests.

Usage:
    uv run python scripts/make_synthetic_dicom.py OUT_DIR [--from-png image.png] [--n 3]

With --from-png, wraps a real PNG (e.g. an NIH ChestX-ray14 image) as a DX DICOM; otherwise draws a
crude phantom (two dark lung fields in a bright body). Phantoms are for plumbing tests only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radio.ingest.synthetic import phantom, write_dx_dicom  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--from-png", type=Path, default=None)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.from_png:
        import numpy as np
        from PIL import Image

        arr = np.asarray(Image.open(a.from_png).convert("L"), dtype=np.uint16) * 16
        p = write_dx_dicom(arr, a.out_dir / (a.from_png.stem + ".dcm"))
        print(p)
        return
    for i in range(a.n):
        print(write_dx_dicom(phantom(seed=i), a.out_dir / f"phantom_{i}.dcm"))


if __name__ == "__main__":
    main()
