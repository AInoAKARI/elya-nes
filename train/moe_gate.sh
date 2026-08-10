#!/bin/sh
# Everything a mixture cartridge has to pass, in one place, in the order that
# makes a silent failure impossible to mistake for a pass:
#   1 pack, and PROVE max|dW| = 0 by decoding the stream back out
#   2 assemble and run the real ROM against the host reference
#   3 show that the survey actually routes to every expert
#   4 the 64-seed survey itself
#   5 the per-stage cycle profile
#   train/moe_gate.sh runs/<arm>.npz [tag]
set -e
cd "$(dirname "$0")/.."
NPZ=$1
TAG=${2:-MOE}
[ -n "$NPZ" ] || { echo "usage: moe_gate.sh <npz> [tag]"; exit 2; }

echo "########## 1. pack + max|dW| ##########"
sh train/build_trained.sh "$NPZ" 1 | tee "out/${TAG}_PACK.txt"

echo
echo "########## 2. ROM vs host, seed token 1 ##########"
python3 tools/run_nn.py out/nn.nes out/model/expected.json 300 \
    | tee "out/${TAG}_VERIFICATION.txt" | tail -25

echo
echo "########## 3. expert coverage of the survey ##########"
python3 train/expert_coverage.py "$NPZ" 64 | tee "out/${TAG}_COVERAGE.txt"

echo
echo "########## 4. 64-seed survey ##########"
sh train/survey_exact.sh "$NPZ" 64 | tee "out/${TAG}_SURVEY.txt" | tail -3

echo
echo "########## 5. cycle profile ##########"
sh train/build_trained.sh "$NPZ" 1 > /dev/null
python3 tools/run_profile.py out/nnprof.nes | tee "out/${TAG}_PROFILE.txt"
