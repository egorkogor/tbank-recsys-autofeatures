Hypotheses, stage val, part val_B; paired bootstrap 1000 over users; Holm over 6 hypotheses

| hypothesis | pair | Δ GAUC [95% CI] | p (Holm) | verdict (Holm) | Δ NDCG@10 [95% CI] | p (Holm) | verdict (Holm) |
|---|---|---|---|---|---|---|---|
| H1 R1 (decay search) is not worse than B0 | r1_decay vs b0 | +0.0047 [+0.0023, +0.0073] | 0.012 | лучше | +0.0045 [+0.0023, +0.0067] | 0.012 | лучше |
| H2 Perseus alone is worse than B0 | r3_perseus vs b0 | -0.0352 [-0.0407, -0.0304] | 0.012 | хуже | -0.0299 [-0.0343, -0.0260] | 0.012 | хуже |
| H3 B0 + Perseus score is better than B0 | b0_perseus vs b0 | +0.0039 [+0.0014, +0.0064] | 0.012 | лучше | +0.0039 [+0.0018, +0.0060] | 0.012 | лучше |
| H4 decay search is not worse than window search at an equal count | r1_decay_k56 vs r1_window_k56 | -0.0049 [-0.0081, -0.0018] | 0.012 | хуже | -0.0041 [-0.0062, -0.0020] | 0.012 | хуже |
| H5 the search beats B0 at an equal count (window grid) | r1_window_k56 vs b0 | +0.0049 [+0.0020, +0.0077] | 0.012 | лучше | +0.0050 [+0.0027, +0.0075] | 0.012 | лучше |
| H7 request-time context: searched beats hand-made | r1_ctx vs b0_ctx | -0.0036 [-0.0081, +0.0007] | 0.112 | не определено | -0.0013 [-0.0039, +0.0012] | 0.338 | сопоставимо |
