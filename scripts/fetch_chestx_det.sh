#!/usr/bin/env bash
# Fetch ChestX-Det (Deepwise; 3,578 NIH ChestX-ray14 images, 13 classes, boxes + polygons; Apache-2.0
# annotations) from the locations given in https://github.com/Deepwise-AILab/ChestX-Det-Dataset.
# Images are served over plain HTTP by the dataset owner, so archives are validated before extraction
# (PNG-only members, no absolute/parent paths) and every PNG is decoded once afterwards.
set -euo pipefail
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/chestx_det"
mkdir -p "$OUT"
cd "$OUT"

for split in train test; do
  json="ChestX_Det_${split}.json"
  [ -s "$json" ] || gh api -H "Accept: application/vnd.github.raw" \
      "repos/Deepwise-AILab/ChestX-Det-Dataset/contents/${json}" > "$json"
done

for split in train test; do
  if [ -d "$split" ] && [ "$(find "$split" -name '*.png' | wc -l)" -gt 0 ]; then echo "skip $split"; continue; fi
  zip="${split}_data.zip"
  curl -fL --retry 3 -o "$zip" "http://resource.deepwise.com/ChestX-Det/${zip}"
  file "$zip" | grep -q "Zip archive" || { echo "not a zip: $zip" >&2; exit 1; }
  bad=$(unzip -Z1 "$zip" | grep -vE '^[A-Za-z0-9_./-]+$|^$' || true)
  trav=$(unzip -Z1 "$zip" | grep -E '(^/|\.\./)' || true)
  nonpng=$(unzip -Z1 "$zip" | grep -vE '(/$|\.png$)' || true)
  if [ -n "$bad$trav$nonpng" ]; then echo "unexpected archive members in $zip:" >&2; echo "$bad$trav$nonpng" | head >&2; exit 1; fi
  mkdir -p "$split"
  unzip -q -j "$zip" '*.png' -d "$split"
  rm -f "$zip"
done
echo "train PNGs: $(find train -name '*.png' | wc -l | tr -d ' ')  test PNGs: $(find test -name '*.png' | wc -l | tr -d ' ')"
du -sh "$OUT"
