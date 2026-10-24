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
    # equal count with B0 (57 aggregates): the first 57 names of the search's ranked list
    "r1_decay_k57": {"b0": BASE_COLS, "r1_decay": ("top", "r1_decay_selected.json", 57)},
    "r1_window_k57": {"b0": BASE_COLS, "r1_window": ("top", "r1_window_selected.json", 57)},
    # model scores as features (stacking), first-stage models trained before the booster's window
    "b0_ials": {"b0": None, "ials_{stage}": None},
    "b0_perseus": {"b0": None, "perseus_{stage}": None},
    "b0_r1": {"b0": None, "r1_decay": None},
    "full": {"b0": None, "r1_decay": None, "ials_{stage}": None, "perseus_{stage}": None},
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
