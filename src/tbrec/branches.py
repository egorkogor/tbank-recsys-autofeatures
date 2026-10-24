"""Which feature files and columns each branch uses. "{stage}" in a file stem is replaced by the stage
(first-stage model scores differ between stages because those models are retrained with the shift)."""

RHYTHM = [f"U_rhythm_daypart_{i}" for i in range(4)] + ["U_rhythm_active_days_30", "U_rhythm_gap_cv_90"]
BASE_COLS = ["I__listen__age__all", "U__listen__age__all", "track_len"] + RHYTHM

BRANCHES = {
    # hand-made baseline
    "b0": {"b0": None},
    # B0 whose user-keyed aggregates see only the last 512 events (the history Perseus sees)
    "b0_trunc512": {"b0_trunc512": None},
    # automatic search over the grammar: same base block, aggregates chosen by the search
    "r1_decay": {"b0": BASE_COLS, "r1_decay": None},
    "r1_window": {"b0": BASE_COLS, "r1_window": None},
    # model scores as features (stacking), first-stage models trained before the booster's window
    "b0_ials": {"b0": None, "ials_{stage}": None},
    "b0_audio": {"b0": None, "audio_{stage}": None},
    "b0_perseus": {"b0": None, "perseus_{stage}": None},
    "b0_r1": {"b0": None, "r1_decay": None},
    "full": {"b0": None, "r1_decay": None, "ials_{stage}": None, "audio_{stage}": None, "perseus_{stage}": None},
}

# Single-score rankers for scale (no training).
TRIVIAL = {
    "trivial_item_lp_rate": ("b0", "I__lp__rate__w30d"),
    "trivial_seen_before": ("b0", "UI__listen__cnt__all"),
    "trivial_user_artist_lp": ("b0", "UA__lp__rate__all"),
}


def resolve(branch: str, stage: str) -> dict:
    return {k.replace("{stage}", stage): v for k, v in BRANCHES[branch].items()}
