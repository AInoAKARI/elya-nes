#!/bin/sh
# Build everything from scratch: model binaries, then every ROM variant.
set -e
cd "$(dirname "$0")"
mkdir -p out out/model

# NES_T is the context length; host/ref.py and rom/nn.s must be given the same
# value or the positional table and the KV cache disagree.  Passing it in one
# place is the whole point.
NES_T=${NES_T:-20}
export NES_T
NCTXDEF="-DNCTX=$NES_T"

python3 host/ref.py out/model > out/model/pack_report.txt
cat out/model/pack_report.txt

build() {   # build <name> <src> <cfg> [defines...]
    name=$1; src=$2; cfg=$3; shift 3
    ca65 -I rom -I out/model "$@" -o "out/$name.o" "$src"
    ld65 -C "$cfg" -o "out/$name.nes" -Ln "out/$name.lbl" "out/$name.o" 2>&1 \
        | grep -v "Segment 'CHARS' does not exist" || true
}

build calib   rom/calib.s rom/nrom.cfg
build prim    rom/prim.s  rom/mmc5.cfg
build mmc1    rom/mmc1.s  rom/mmc1.cfg
build mmc3    rom/mmc3.s  rom/mmc3.cfg
build nn      rom/nn.s    rom/nn.cfg $NCTXDEF
build nnprof  rom/nn.s    rom/nn.cfg $NCTXDEF -DPROFILE
build nnbench rom/nn.s    rom/nn.cfg $NCTXDEF -DBENCH
if [ "$NES_T" -le 20 ]; then
    build nndbg   rom/nn.s    rom/nn.cfg $NCTXDEF -DDEBUG -DDBGPOS=0
fi

ls -l out/*.nes
