#!/bin/sh
# Take a trained npz all the way to a verified cartridge.
#   train/build_trained.sh runs/<arm>.npz [seed_tok]
# Order matters: pack, PROVE max|dW| = 0, then assemble.  Assembling first
# would mean a silent exporter bug becomes a ROM that runs perfectly and says
# the wrong thing, which is exactly how the sibling N64 port lost a week.
set -e
cd "$(dirname "$0")/.."
NPZ=$1
SEED=${2:-1}
[ -n "$NPZ" ] || { echo "usage: build_trained.sh <npz> [seed_tok]"; exit 2; }

mkdir -p out out/model
NES_T=${NES_T:-20}
export NES_T
# A denser model needs more than the seven 8 KB weight-stream banks.  The
# packer, the assembler and the linker config all have to agree about how many,
# so it is set in ONE place here.
NES_STREAM_BANKS=${NES_STREAM_BANKS:-7}
export NES_STREAM_BANKS
if [ "$NES_STREAM_BANKS" -eq 7 ]; then CFG=rom/nn.cfg
elif [ "$NES_STREAM_BANKS" -eq 9 ]; then CFG=rom/nn9.cfg
else echo "no linker config for $NES_STREAM_BANKS stream banks"; exit 2; fi
NES_WEIGHTS="$NPZ" NES_SEED_TOK="$SEED" \
    python3 host/ref.py out/model | tee out/model/pack_report.txt

echo
python3 train/verify_pack.py "$NPZ" --dir out/model | tail -6

echo
build() {
    name=$1; src=$2; cfg=$3; shift 3
    ca65 -I rom -I out/model "$@" -o "out/$name.o" "$src"
    ld65 -C "$cfg" -o "out/$name.nes" -Ln "out/$name.lbl" "out/$name.o" 2>&1 \
        | grep -v "Segment 'CHARS' does not exist" || true
}
DEFS="-DNCTX=$NES_T -DNSTREAM=$NES_STREAM_BANKS -DSEEDTOK=$SEED"
build nn      rom/nn.s $CFG $DEFS
build nnprof  rom/nn.s $CFG $DEFS -DPROFILE
ls -l out/nn.nes out/nnprof.nes
