#!/usr/bin/env bash
# Fetch a subset of NIH ChestX-ray14 from the official NIH Clinical Center Box share.
#   https://nihcc.app.box.com/v/ChestXray-NIHCC
# Terms (NIH README/FAQ): usage unrestricted; link to the NIH site, acknowledge the NIH Clinical Center,
# cite Wang et al. CVPR 2017. Images are NOT redistributed by this repo (data/ is git-ignored).
#
# Usage: scripts/fetch_nih_subset.sh [N_BATCHES=1]   (each batch ~2-4 GB; batch 1 is 1.9 GB)
set -euo pipefail

N=${1:-1}
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/nih"
mkdir -p "$OUT/images"
cd "$OUT"

BOX="https://nihcc.app.box.com/index.php?rm=box_download_shared_file&vanity_name=ChestXray-NIHCC&file_id=f_"
declare -a META=(
  "Data_Entry_2017_v2020.csv:219760887468"
  "train_val_list.txt:256056636701"
  "test_list.txt:256055473534"
  "BBox_List_2017.csv:219760940956"
)
for entry in "${META[@]}"; do
  name="${entry%%:*}"; id="${entry##*:}"
  [ -s "$name" ] || curl -fL --retry 3 -o "$name" "${BOX}${id}"
done
ln -sf Data_Entry_2017_v2020.csv Data_Entry_2017.csv

# Image archive URLs exactly as listed in NIH's batch_download_zips.py (Box file 371647823217).
LINKS=(
  https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz
  https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz
  https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz
  https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz
  https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz
  https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz
  https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz
  https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz
  https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz
  https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz
  https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz
  https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz
)
for ((i = 0; i < N; i++)); do
  fn=$(printf "images_%03d.tar.gz" $((i + 1)))
  dest="images/$(printf "images_%03d" $((i + 1)))"
  if [ -d "$dest" ]; then echo "skip $dest"; continue; fi
  echo "downloading $fn"
  curl -fL --retry 3 -o "$fn" "${LINKS[$i]}"
  file "$fn" | grep -q gzip || { echo "not a gzip archive: $fn" >&2; exit 1; }
  mkdir -p "$dest"
  tar -xzf "$fn" -C "$dest"
  rm -f "$fn"
done
echo "PNG count: $(find images -name '*.png' | wc -l | tr -d ' ')"
du -sh "$OUT"
