from __future__ import annotations

from datetime import timedelta

from services.models.features import (
    Tier,
    combine_corpora,
    importance_tier,
    normalize_team,
    team_name_id,
    with_name_ids,
)
from tests.conftest import KICKOFF_BASE, MatchFactory


def test_normalize_folds_case_diacritics_and_aliases() -> None:
    assert normalize_team("  USA ") == "united states"
    assert normalize_team("United States") == "united states"
    assert normalize_team("Korea Republic") == "south korea"
    assert normalize_team("Côte d'Ivoire") == "ivory coast"
    assert normalize_team("Türkiye") == "turkey"
    assert normalize_team("Curaçao") == "curacao"
    assert normalize_team("Brazil") == "brazil"


def test_name_ids_are_stable_and_alias_invariant() -> None:
    assert team_name_id("USA") == team_name_id("United States")
    assert team_name_id("Brazil") == team_name_id("brazil")
    assert team_name_id("Brazil") != team_name_id("Argentina")


def test_with_name_ids_preserves_fixture_id(make_match: MatchFactory) -> None:
    match = make_match(id=777, home="USA", away="Mexico", home_id=64, away_id=65)
    rekeyed = with_name_ids(match)
    assert rekeyed.id == 777
    assert rekeyed.home_id == team_name_id("United States")
    assert rekeyed.away_id == team_name_id("Mexico")


def test_importance_tiers() -> None:
    assert importance_tier("Friendly") is Tier.FRIENDLY
    assert importance_tier("FIFA World Cup qualification") is Tier.QUALIFIER
    assert importance_tier("UEFA Nations League") is Tier.QUALIFIER
    assert importance_tier("UEFA Euro qualification") is Tier.QUALIFIER
    assert importance_tier("UEFA Euro") is Tier.CONTINENTAL
    assert importance_tier("Copa América") is Tier.CONTINENTAL
    assert importance_tier("FIFA World Cup") is Tier.WC_FINALS
    assert importance_tier("GROUP_STAGE") is Tier.WC_FINALS
    assert importance_tier("LAST_32") is Tier.WC_FINALS
    assert importance_tier("QUARTER_FINALS") is Tier.WC_FINALS
    assert importance_tier("Some Minor Cup") is Tier.MINOR


def test_combine_corpora_dedupes_preferring_provider_row(
    make_match: MatchFactory,
) -> None:
    # same real-world match: CSV row at noon, provider row with the real kickoff
    history_row = make_match(
        id=111,
        kickoff=KICKOFF_BASE.replace(hour=12, minute=0),
        home="USA",
        away="Mexico",
        home_goals=1,
        away_goals=0,
        stage="FIFA World Cup",
    )
    provider_row = make_match(
        id=222,
        kickoff=KICKOFF_BASE.replace(hour=20, minute=0),
        home="United States",
        away="Mexico",
        home_id=64,
        away_id=65,
        home_goals=1,
        away_goals=0,
        stage="GROUP_STAGE",
    )
    other = make_match(
        id=333,
        kickoff=KICKOFF_BASE - timedelta(days=30),
        home="Brazil",
        away="Argentina",
        home_goals=2,
        away_goals=2,
        stage="Friendly",  # history rows carry tournament names, not provider tokens
    )

    combined = combine_corpora([history_row, other], [provider_row])
    assert [m.id for m in combined] == [333, 222]
    usa_mexico = combined[1]
    assert usa_mexico.home_id == team_name_id("USA")
    assert usa_mexico.utc_kickoff.hour == 20


def test_combine_corpora_drops_history_finals_twin_across_utc_date_boundary(
    make_match: MatchFactory,
) -> None:
    # same real-world match, 20:00 PDT kickoff: the history CSV stamps it noon
    # UTC on the LOCAL date, the provider row lands at 03:00 UTC the NEXT day —
    # different UTC dates, so date-keyed dedup alone would keep both (and here
    # they even disagree on the result). The provider is authoritative for the
    # finals seasons it covers; the history twin must not survive as an earlier-
    # timestamped leak of the same result.
    history_twin = make_match(
        id=444,
        kickoff=KICKOFF_BASE.replace(hour=12, minute=0),  # noon UTC, June 15
        season=2026,
        home="USA",
        away="Mexico",
        home_goals=1,
        away_goals=1,
        stage="FIFA World Cup",
    )
    provider_row = make_match(
        id=555,
        kickoff=KICKOFF_BASE.replace(hour=3, minute=0) + timedelta(days=1),
        season=2026,
        home="United States",
        away="Mexico",
        home_goals=3,
        away_goals=4,
        stage="LAST_32",
    )
    past_finals = make_match(  # provider does NOT cover 2018: row must survive
        id=666,
        kickoff=KICKOFF_BASE.replace(year=2018) - timedelta(days=8 * 365),
        season=2018,
        home="France",
        away="Croatia",
        home_goals=4,
        away_goals=2,
        stage="FIFA World Cup",
    )

    combined = combine_corpora([history_twin, past_finals], [provider_row])
    assert [m.id for m in combined] == [666, 555]  # twin dropped, 2018 row kept
    assert combined[1].home_goals == 3  # the provider result, not the twin's

    # without provider coverage the history finals rows remain the sole source
    history_only = combine_corpora([history_twin, past_finals], [])
    assert [m.id for m in history_only] == [666, 444]
