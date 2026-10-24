"""Hand-made feature set of the baseline B0 (fixed windows 1 / 7 / 30 days and all time).

Chosen the way an engineer would: activity and quality of the user, popularity and quality of the
track, direct user x track history, indirect user x artist / album history. BASE is shared by every
branch: static content and lifetimes, not aggregates whose windows are being searched.
"""

BASE = ["I__listen__age__all", "U__listen__age__all"]  # + track_len, rhythm, audio cosine (added in build)

B0 = [
    # user
    "U__listen__cnt__w1d", "U__listen__cnt__w7d", "U__listen__cnt__w30d", "U__listen__cnt__all",
    "U__lp__rate__w7d", "U__lp__rate__w30d", "U__lp__rate__all",
    "U__skip__rate__w30d", "U__skip__rate__all", "U__org__rate__w30d", "U__org__rate__all",
    "U__like__cnt__w30d", "U__like__cnt__all", "U__dislike__cnt__all",
    "U__listen__nuniq__w30d", "U__listen__mpr__w30d", "U__listen__rec__all",
    # item
    "I__listen__cnt__w1d", "I__listen__cnt__w7d", "I__listen__cnt__w30d", "I__listen__cnt__all",
    "I__lp__rate__w7d", "I__lp__rate__w30d", "I__lp__rate__all",
    "I__skip__rate__w30d", "I__algo__rate__w30d", "I__listen__mpr__w30d",
    "I__like__cnt__w30d", "I__like__cnt__all", "I__like__rate__all", "I__dislike__cnt__all",
    # artist / album
    "A__listen__cnt__w7d", "A__listen__cnt__w30d", "A__lp__rate__w30d", "A__like__cnt__all",
    "L__listen__cnt__w30d", "L__lp__rate__w30d",
    # user x item (direct)
    "UI__listen__cnt__w30d", "UI__listen__cnt__all", "UI__lp__cnt__all", "UI__lp__rate__all",
    "UI__skip__cnt__all", "UI__listen__mpr__all", "UI__listen__rec__all",
    "UI__like__cnt__all", "UI__dislike__cnt__all",
    # user x artist / album (indirect)
    "UA__listen__cnt__w30d", "UA__listen__cnt__all", "UA__lp__rate__all", "UA__skip__rate__all",
    "UA__like__cnt__all", "UA__dislike__cnt__all", "UA__listen__rec__all",
    "UL__listen__cnt__all", "UL__lp__rate__all", "UL__listen__rec__all",
]
