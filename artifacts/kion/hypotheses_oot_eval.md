Hypotheses, stage oot, part eval; paired bootstrap 1000 over users; Holm over 3 hypotheses

| hypothesis | pair | Δ GAUC [95% CI] | p (Holm) | verdict (Holm) | Δ NDCG@10 [95% CI] | p (Holm) | verdict (Holm) |
|---|---|---|---|---|---|---|---|
| H1 R1 (decay search) is not worse than B0 | r1_decay vs b0 | -0.0019 [-0.0055, +0.0016] | 0.627 | сопоставимо | +0.0086 [+0.0019, +0.0150] | 0.030 | лучше |
| H4 decay search is not worse than window search at an equal count | r1_decay_k33 vs r1_window_k33 | -0.0004 [-0.0030, +0.0024] | 0.789 | сопоставимо | +0.0049 [+0.0002, +0.0099] | 0.088 | не определено (после Холма) |
| H5 the search beats B0 at an equal count (window grid) | r1_window_k33 vs b0 | -0.0030 [-0.0065, +0.0007] | 0.330 | не определено | +0.0012 [-0.0049, +0.0077] | 0.657 | сопоставимо |
