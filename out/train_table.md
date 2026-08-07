| arm | vocab | quant | tau | fit | val | val/char | density | nnz | banks | fits 7? |
|---|---|---|---|---|---|---|---|---|---|---|
| bpe64_twn_tau1.00_q1_s1 | bpe64 | float W | 1.00 | 2.2044 | 2.2075 | **1.5185** | 0.4256 | 43583 | 6 | yes |
| bpe64_twn_tau0.75_q2_s1 | bpe64 | QAT | 0.75 | 2.2401 | 2.2435 | **1.5432** | 0.5441 | 55711 | 7 | yes |
| bpe64_bn_tau1.00_q2_s1 | bpe64 | QAT | 1.00 | 2.2566 | 2.2586 | **1.5537** | 0.6738 | 69002 | 9 | **NO** |
| bpe64_twn_tau0.50_q2_s1 | bpe64 | QAT | 0.50 | 2.2566 | 2.2586 | **1.5537** | 0.6738 | 69002 | 9 | **NO** |
| bpe64_twn_tau1.00_q2_s1 | bpe64 | QAT | 1.00 | 2.2824 | 2.2874 | **1.5734** | 0.4137 | 42360 | 6 | yes |
| bpe64_twn_tau1.00_q2_s2 | bpe64 | QAT | 1.00 | 2.2984 | 2.3008 | **1.5827** | 0.4148 | 42475 | 6 | yes |
| bpe64_twn_tau1.25_q2_s1 | bpe64 | QAT | 1.25 | 2.3823 | 2.3855 | **1.6410** | 0.3080 | 31538 | 4 | yes |
| charset_twn_tau1.00_q2_s1 | charset | QAT | 1.00 | 1.7041 | 1.6999 | **1.6999** | 0.4224 | 43257 | 6 | yes |
| bpe64_twn_tau1.50_q2_s1 | bpe64 | QAT | 1.50 | 2.5246 | 2.5290 | **1.7396** | 0.2171 | 22230 | 3 | yes |
| bpe64_twn_tau1.00_q0_s1 | bpe64 | fp32 | 1.00 | 2.6446 | 2.6472 | **1.8209** | 0.4249 | 43506 | 6 | yes |

uniform baseline ln(64) = 4.1589
7-bank stream window holds at most 57232 index bytes -> density 0.5589
