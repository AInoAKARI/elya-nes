# Sources / Claim Map

Every factual claim in this package is tied to public repository or issue evidence.

## Project framing

**Claim:** `elya-nes` is a cycle-exact NES/Famicom 2A03 measurement instrument with a ternary transformer forward pass verified against a host reference.

- Upstream README: https://github.com/Scottcjn/elya-nes/blob/main/README.md
- Measured source commit used by the reproduction: https://github.com/Scottcjn/elya-nes/commit/2de04401cd7de09d9c44712d3eb4b2dd7190518c

## Independent Mesen reproduction

**Claim:** The reproduction used Mesen 2.1.1 and a separate harness that records CPU/master-clock timing markers.

- Upstream PR #3: https://github.com/Scottcjn/elya-nes/pull/3
- Reproduction commit: https://github.com/AInoAKARI/elya-nes/commit/71d89c6865607bd222e952133cf1216514d31b17
- Claim record: https://github.com/Scottcjn/rustchain-bounties/issues/16582

## Token identity

**Claim:** Host expected output and Mesen observed output were byte-identical for all 19 generated tokens.

- Raw result packet: https://github.com/AInoAKARI/elya-nes/blob/bounty/16517-mesen/results/nes_mesen_2.1.1_dense_pow2.json
- PR #3 result summary: https://github.com/Scottcjn/elya-nes/pull/3
- Claim #16582: https://github.com/Scottcjn/rustchain-bounties/issues/16582

## Cycle counts

**Claim:** 21,227,718 total CPU cycles; 1,117,248 mean CPU cycles/token; 254,732,616 total NES master clocks.

- Raw result packet: https://github.com/AInoAKARI/elya-nes/blob/bounty/16517-mesen/results/nes_mesen_2.1.1_dense_pow2.json
- PR #3: https://github.com/Scottcjn/elya-nes/pull/3

## Clock ratio

**Claim:** Every measured NES master-clock delta was exactly 12× its CPU-cycle delta.

- Raw result packet: https://github.com/AInoAKARI/elya-nes/blob/bounty/16517-mesen/results/nes_mesen_2.1.1_dense_pow2.json
- PR #3: https://github.com/Scottcjn/elya-nes/pull/3

## Cross-emulator comparison

**Claim:** The Mesen mean was 1,117,248 CPU cycles/token, a 0-cycle (0.0%) difference from the published MAME mean used for this bounty reproduction.

- Mesen claim: https://github.com/Scottcjn/rustchain-bounties/issues/16582
- Prior MAME reproduction claim: https://github.com/Scottcjn/rustchain-bounties/issues/16574
- Mesen PR #3: https://github.com/Scottcjn/elya-nes/pull/3

## Production safety

The storyboard requires only public repository captures, original text/motion graphics, and optional original sound effects. It does not require Nintendo game footage, copyrighted music, private credentials, or local secret-bearing UI.
