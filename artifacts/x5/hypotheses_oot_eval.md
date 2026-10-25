Hypotheses, stage oot, part eval; paired bootstrap 1000 over users; Holm over 3 hypotheses

| hypothesis | pair | Δ GAUC [95% CI] | p (Holm) | verdict (Holm) | Δ NDCG@10 [95% CI] | p (Holm) | verdict (Holm) |
|---|---|---|---|---|---|---|---|
| H1 R1 (decay search) is not worse than B0 | r1_decay vs b0 | -0.0004 [-0.0014, +0.0004] | 0.791 | сопоставимо | -0.0016 [-0.0037, +0.0005] | 0.264 | сопоставимо |
| H4 decay search is not worse than window search at an equal count | r1_decay_k25 vs r1_window_k25 | -0.0013 [-0.0022, -0.0005] | 0.012 | хуже | -0.0028 [-0.0048, -0.0009] | 0.018 | хуже |
| H5 the search beats B0 at an equal count (window grid) | r1_window_k25 vs b0 | -0.0002 [-0.0011, +0.0008] | 0.791 | сопоставимо | +0.0005 [-0.0016, +0.0026] | 0.615 | сопоставимо |
