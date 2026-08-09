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
        | grep -v "Segment 'CHARS' does not exist" \
        | grep -v "Segment 'POS' does not exist" || true
}
# 'POS' is empty whenever the positional table shares the embedding bank,
# which is every build at the default context length.  The memory area is
# still filled, so the image size does not move.

build calib   rom/calib.s rom/nrom.cfg
build prim    rom/prim.s  rom/mmc5.cfg
build mmc1    rom/mmc1.s  rom/mmc1.cfg
build mmc3    rom/mmc3.s  rom/mmc3.cfg
build nn      rom/nn.s    rom/nn.cfg $NCTXDEF
build nnprof  rom/nn.s    rom/nn.cfg $NCTXDEF -DPROFILE
build nnbench rom/nn.s    rom/nn.cfg $NCTXDEF -DBENCH
# The DEBUG snapshots and the three attention instruments only exist while the
# attention kernels do, i.e. NCTX <= 64/L = 21.  Above that the ROM is on the
# legacy attention path and these targets have nothing to measure; nn.s
# refuses to assemble them rather than emit something that looks right.
if [ "$NES_T" -le 21 ]; then
    build nndbg   rom/nn.s    rom/nn.cfg $NCTXDEF -DDEBUG -DDBGPOS=0
    build nnattn  rom/nn.s    rom/nn.cfg $NCTXDEF -DATTNPROF   # attention breakdown
    build nnabench rom/nn.s   rom/nn.cfg $NCTXDEF -DATTNBENCH  # attention kernel slope
    build ramexec rom/nn.s    rom/nn.cfg $NCTXDEF -DRAMEXEC    # MMC5 PRG-RAM probe
fi

# The bank-budget probe is a 1 MB image and is built last so that a failure
# here cannot be mistaken for a failure of the cartridge itself.
python3 tools/gen_bankstamp.py 127 out/model/bankstamp.bin > /dev/null
ca65 -I rom -I out/model -o out/bankprobe.o rom/bankprobe.s
ld65 -C rom/bankprobe.cfg -o out/bankprobe.nes out/bankprobe.o

ls -l out/*.nes
