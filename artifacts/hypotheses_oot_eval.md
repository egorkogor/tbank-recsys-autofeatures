Hypotheses, stage oot, part eval; paired bootstrap 1000 over users; Holm over 6 hypotheses

| hypothesis | pair | Δ GAUC [95% CI] | p (Holm) | verdict (Holm) | Δ NDCG@10 [95% CI] | p (Holm) | verdict (Holm) |
|---|---|---|---|---|---|---|---|
| H1 R1 (decay search) is not worse than B0 | r1_decay vs b0 | +0.0050 [+0.0030, +0.0069] | 0.012 | лучше | +0.0048 [+0.0034, +0.0063] | 0.012 | лучше |
| H2 Perseus alone is worse than B0 | r3_perseus vs b0 | -0.0331 [-0.0371, -0.0293] | 0.012 | хуже | -0.0277 [-0.0307, -0.0251] | 0.012 | хуже |
| H3 B0 + Perseus score is better than B0 | b0_perseus vs b0 | +0.0037 [+0.0015, +0.0060] | 0.012 | лучше | +0.0047 [+0.0033, +0.0063] | 0.012 | лучше |
| H4 decay search is not worse than window search at an equal count | r1_decay_k56 vs r1_window_k56 | -0.0053 [-0.0072, -0.0034] | 0.012 | хуже | -0.0038 [-0.0052, -0.0023] | 0.012 | хуже |
| H5 the search beats B0 at an equal count (window grid) | r1_window_k56 vs b0 | +0.0060 [+0.0039, +0.0081] | 0.012 | лучше | +0.0057 [+0.0040, +0.0072] | 0.012 | лучше |
| H7 request-time context: searched beats hand-made | r1_ctx vs b0_ctx | -0.0069 [-0.0101, -0.0037] | 0.012 | хуже | +0.0004 [-0.0015, +0.0021] | 0.691 | сопоставимо |
