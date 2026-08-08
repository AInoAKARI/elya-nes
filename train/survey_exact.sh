#!/bin/sh
# ROM == host over every generated token, at many seed tokens.
#   train/survey_exact.sh <npz> [nseeds]
# One seed is a lucky-start check; this is the version that is evidence.
set -e
cd "$(dirname "$0")/.."
NPZ=$1
N=${2:-16}
# The context length has to reach the packer (NES_T), the assembler (-DNCTX)
# and the token count in one move, or the three disagree silently.
NES_T=${NES_T:-20}
export NES_T
NTOK=$((NES_T - 1))
S=0
PASS=0
while [ "$S" -lt "$N" ]; do
    NES_WEIGHTS="$NPZ" NES_SEED_TOK="$S" python3 host/ref.py out/model "$NTOK" >/dev/null
    ca65 -I rom -I out/model -DNCTX=$NES_T -DSEEDTOK=$S -o out/sv.o rom/nn.s
    ld65 -C rom/nn.cfg -o out/sv.nes out/sv.o 2>&1 \
        | grep -v "CHARS" | grep -v "Segment 'POS' does not exist" || true
    R=$(python3 tools/run_nn.py out/sv.nes out/model/expected.json 300 | grep "TOKENS MATCHING")
    echo "seed $S: $R"
    case "$R" in *EXACT*) PASS=$((PASS+1));; esac
    S=$((S+1))
done
echo "-----------------------------------------"
echo "seeds exact: $PASS / $N   ($((PASS*NTOK)) / $((N*NTOK)) tokens)"
