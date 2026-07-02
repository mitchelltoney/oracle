"""Feature engineering — reads only versioned snapshots (hard rule 2)."""

from services.models.features.corpus import Tier, combine_corpora, importance_tier
from services.models.features.team_names import (
    ALIASES,
    HOST_NATIONS_BY_SEASON,
    normalize_team,
    team_name_id,
    with_name_ids,
)

__all__ = [
    "ALIASES",
    "HOST_NATIONS_BY_SEASON",
    "Tier",
    "combine_corpora",
    "importance_tier",
    "normalize_team",
    "team_name_id",
    "with_name_ids",
]
