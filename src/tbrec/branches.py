"""Which feature files and columns each branch uses. "{stage}" in a file stem is replaced by the stage
(first-stage model scores differ between stages because those models are retrained with the shift)."""

RHYTHM = [f"U_rhythm_daypart_{i}" for i in range(4)] + ["U_rhythm_active_days_30", "U_rhythm_gap_cv_90"]
BASE_COLS = ["I__listen__age__all", "U__listen__age__all", "track_len"] + RHYTHM

BRANCHES = {
    # hand-made baseline
    "b0": {"b0": None},
    # B0 whose user-keyed aggregates see only the user's last N events (the history Perseus sees)
    "b0_trunc256": {"b0_trunc256": None},
    "b0_trunc512": {"b0_trunc512": None},
    # automatic search over the grammar: same base block, aggregates chosen by the search
    "r1_decay": {"b0": BASE_COLS, "r1_decay": None},
    "r1_window": {"b0": BASE_COLS, "r1_window": None},
    # R1 + features proposed by the agent (R5) that passed the leak check and the val-A gain threshold
    "r1_r5": {"b0": BASE_COLS, "r1_decay": None, "r5": None},
    # R1 with the stability stability gate (top on >= 3 of 5 separate training days)
    "r1_decay_stable": {"b0": BASE_COLS, "r1_decay_stable": None},
    # equal count with B0 (56 aggregates): the first 56 names of the search's ranked list
    "r1_decay_k56": {"b0": BASE_COLS, "r1_decay": ("top", "r1_decay_selected.json", 56)},
    "r1_window_k56": {"b0": BASE_COLS, "r1_window": ("top", "r1_window_selected.json", 56)},
    # model scores as features (stacking), first-stage models trained before the booster's window
    "b0_ials": {"b0": None, "ials_{stage}": None},
    "b0_perseus": {"b0": None, "perseus_{stage}": None},
    # audio content (no training: snapshot of the user's taste profile, stage-independent)
    "b0_audio": {"b0": None, "audio": None},
    "b0_r1": {"b0": None, "r1_decay": None},
    # request-time context (hour / part of day / weekday of the impression): hand-made vs searched
    "b0_ctx": {"b0": None, "ctx_b0": None},
    "r1_ctx": {"b0": BASE_COLS, "r1_decay": None, "ctx_r1": None},
    # listener-state proxies on top of the hand context: sound x time of day, yesterday's sound, audio clusters
    "b0_ctx_mood": {"b0": None, "ctx_b0": None, "mood": None},
    "full": {"b0": None, "r1_decay": None, "ials_{stage}": None, "audio": None, "perseus_{stage}": None},
}

# Single-score rankers for scale (no training).
TRIVIAL = {
    "trivial_item_lp_rate": ("b0", "I__lp__rate__w30d"),
    "trivial_seen_before": ("b0", "UI__listen__cnt__all"),
    "trivial_user_artist_lp": ("b0", "UA__lp__rate__all"),
}


def resolve(branch: str, stage: str) -> dict:
    import json

    from .config import ART

    out = {}
    for k, v in BRANCHES[branch].items():
        if isinstance(v, tuple):  # ("top", selection file, n): first n names of a search result
            v = json.loads((ART / v[1]).read_text())["names"][: v[2]]
        out[k.replace("{stage}", stage)] = v
    return out


# ---- KION (TBREC_DATASET=kion): same branch names, the dataset's own base block and B0 ----
from .config import DATASET  # noqa: E402

if DATASET in ("kion", "x5"):
    if DATASET == "kion":
        from .kion import B0 as _KB0, BASE as _KBASE, STATIC as _KSTATIC
    else:
        from .x5 import B0 as _KB0, BASE as _KBASE, STATIC as _KSTATIC

    BASE_COLS = _KSTATIC + _KBASE
    BRANCHES = {
        "b0": {"b0": None},
        "r1_decay": {"b0": BASE_COLS, "r1_decay": None},
        "r1_window": {"b0": BASE_COLS, "r1_window": None},
        f"r1_decay_k{len(_KB0)}": {"b0": BASE_COLS, "r1_decay": ("top", "r1_decay_selected.json", len(_KB0))},
        f"r1_window_k{len(_KB0)}": {"b0": BASE_COLS, "r1_window": ("top", "r1_window_selected.json", len(_KB0))},
        "base_only": {"b0": BASE_COLS},
    }
    # candidate-generator ranks: lower is better, so the score is the negated rank
    # iALS score exists for every candidate (the rank only for its own top-50, so it is not a fair ranker)
    TRIVIAL = {"trivial_ials_score": ("b0", "cand_ials_score"), "trivial_pop_rank": ("b0", "-cand_pop_rank")}
    if DATASET == "x5":
        TRIVIAL["trivial_hist_rank"] = ("b0", "-cand_hist_rank")  # "buy again what you buy most"
