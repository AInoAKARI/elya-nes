#!/bin/sh
# Full verification of a trained cartridge (set NES_T for a non-default context): pack, prove max|dW| = 0, assemble,
# then run the real ROM in MAME and diff every generated token against the
# host reference.  Repeated at several seed tokens, because a single starting
# point is exactly the kind of check that has passed broken code in this
# project before.
#   train/verify_trained.sh runs/<arm>.npz "1 26 40"
set -e
cd "$(dirname "$0")/.."
NPZ=$1
SEEDS=${2:-"1 26 40"}
[ -n "$NPZ" ] || { echo "usage: verify_trained.sh <npz> [\"seed seed ...\"]"; exit 2; }

for S in $SEEDS; do
    echo "############################################################"
    echo "# seed token $S"
    echo "############################################################"
    sh train/build_trained.sh "$NPZ" "$S"
    python3 tools/run_nn.py out/nn.nes out/model/expected.json 300
    echo
done
