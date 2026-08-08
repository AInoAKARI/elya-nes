#!/bin/sh
# Build everything from scratch: model binaries, then every ROM variant.
set -e
cd "$(dirname "$0")"
mkdir -p out out/model

python3 host/ref.py out/model 19 > out/model/pack_report.txt
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
build nn      rom/nn.s    rom/nn.cfg
build nnprof  rom/nn.s    rom/nn.cfg -DPROFILE
build nnbench rom/nn.s    rom/nn.cfg -DBENCH
build nndbg   rom/nn.s    rom/nn.cfg -DDEBUG -DDBGPOS=0
build nnattn  rom/nn.s    rom/nn.cfg -DATTNPROF     # attention breakdown
build nnabench rom/nn.s   rom/nn.cfg -DATTNBENCH    # attention kernel slope
build ramexec rom/nn.s    rom/nn.cfg -DRAMEXEC      # MMC5 PRG-RAM probe

ls -l out/*.nes
