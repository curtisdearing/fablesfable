"""Real-source example: factor evidence panel for ATL @ GB, 2026 Week 3 (2026_03_ATL_GB).

Each record is transcribed from a retrieved source.  Where a source could not
be confirmed from a primary body, the record is labelled unverified, not
filled in.

Primary bodies (raw HTML fetched 2026-09-23T01:39Z with a generic User-Agent;
the publish clock is the page's JSON-LD ``datePublished``):

* atlantafalcons.com "Michael Penix Jr. named starting QB for Thursday Night
  Football vs. Packers" -- datePublished 2026-09-21T20:29:04Z.
* packers.com "Packers-Falcons Injury Report | Sept. 21, 2026" -- datePublished
  2026-09-21T21:07:43Z.  Monday participation is an ESTIMATE (Packers did not
  practice; Falcons held a walkthrough).  The page links a Sept. 22 report that
  was not fetched, so these rows may be superseded.
* atlantafalcons.com/schedule/ -- Wk3 Thu 8:15 PM EDT Prime Video, Wk4 Mon
  8:15 PM EDT ESPN, Wk5 Sun 8:20 PM EDT NBC; agrees with nflverse games.csv.

Unverified on purpose:

* A.J. Terrell "injured reserve": only a related-content headline on the
  Falcons site; no publication time was retrieved, so it is not used and is
  not treated as contradicting the Sept. 21 estimate.
* Josh Jacobs "commissioner's exempt list": third-party summary only.

Data rows: PFR snap counts via nflreadpy and nflverse play-by-play (2026 REG
Weeks 1-2, plus the era split table built from play-by-play); NWS hourly
forecast GRB/78,31, updateTime 2026-09-22T17:36:16Z.  Player ids are the gsis
ids on those rows.

No Week 3 production run existed when this was written, so there is no run
receipt.  Model-stage records are therefore "not verified", even where the
code configuration names a shadow path: configuration is not execution.

Run:  python -m analysis.factor_evidence_examples  (prints panel JSON; writes nothing)
"""

from __future__ import annotations

import datetime as dt
import json

from nflvalue import factor_evidence as fe

AS_OF = dt.datetime(2026, 9, 23, 1, 45, tzinfo=dt.timezone.utc)
GAME = "2026_03_ATL_GB"
#: gsis ids as they appear on the nflverse/PFR rows used below
PLAYER_IDS = {"Michael Penix Jr.": "00-0039917", "Josh Jacobs": "00-0035700",
              "Jordan Love": "00-0036264", "Jayden Reed": "00-0039146",
              "Matthew Golden": "00-0040667"}
PENIX, JACOBS = PLAYER_IDS["Michael Penix Jr."], PLAYER_IDS["Josh Jacobs"]

_PENIX_URL = ("https://www.atlantafalcons.com/news/"
              "michael-penix-jr-starting-qb-thursday-night-football-vs-packers")
_GB_REPORT = dict(attribution="Green Bay Packers (joint injury report)", source_tier="team_official",
                  claim_kind="confirmed",
                  source_url="https://www.packers.com/news/packers-falcons-injury-report-sept-21-2026",
                  source_title="Packers-Falcons Injury Report | Sept. 21, 2026",
                  published_at="2026-09-21T21:07:43Z", fetched_at="2026-09-23T01:39:08Z")

NEWS = [
    dict(story_id="penix_starter", entity_id=PENIX, entity_type="player", team="ATL", game_id=GAME,
         category="qb_news", claim_key="starter", claim_value="starting",
         claim="Michael Penix Jr. has been named the Falcons' starting quarterback for the "
               "Thursday night game against the Green Bay Packers.",
         attribution="Atlanta Falcons (team site)", source_tier="team_official", claim_kind="confirmed",
         source_url=_PENIX_URL,
         source_title="Michael Penix Jr. named starting QB for Thursday Night Football vs. Packers",
         published_at="2026-09-21T20:29:04Z", fetched_at="2026-09-23T01:39:07Z",
         rationale="Return from a season-ending knee injury last season (per the article)."),
    dict(story_id="penix_starter_nfl_com", entity_id=PENIX, entity_type="player", team="ATL",
         game_id=GAME, category="qb_news", claim_key="starter", claim_value="starting",
         claim="Falcons starting Michael Penix Jr. vs Packers (search result title).",
         attribution="NFL.com", source_tier="media", claim_kind="report",
         source_url="https://www.nfl.com/news/falcons-michael-penix-jr-starting-quarterback-week-3-game-vs-packers",
         source_title="NFL.com search result", published_at=None, fetched_at="2026-09-22T18:35:00Z"),
    dict(story_id="jacobs_exempt_list", entity_id=JACOBS, entity_type="player", team="GB",
         game_id=GAME, category="team_news", claim_key="availability", claim_value="exempt_list",
         claim="Claimed to be on the commissioner's exempt list.",
         attribution="third-party search summary", source_tier="third_party_summary",
         claim_kind="report", source_url="https://www.fantasypros.com/nfl/myplaybook/are-they-playing/josh-jacobs",
         published_at=None, fetched_at="2026-09-22T18:50:00Z",
         rationale="Observed: 0 offensive snaps in Weeks 1-2 (PFR). Not on the Sept. 21 report."),
    dict(_GB_REPORT, story_id="gb_injury_report_0921_ol", entity_id="GB", entity_type="team", team="GB",
         game_id=GAME, category="ol_injury", claim_key="ol_practice", claim_value="3 OL did not participate",
         claim="Did Not Participate (estimated): Zach Bako-Bewele (knee), Aaron Banks (knee/toe), "
               "Donovan Jennings (hand).",
         rationale="Offensive snap share (PFR), Wk1 -> Wk2: Bako-Bewele 100% -> 5%, Jennings 100% -> 24%; "
                   "Banks 37% in Wk2. No pressure or pass-block data is ingested.",
         uncertainty="Estimated participation (no practice held); a Sept. 22 report exists and was "
                     "not fetched, so this may be superseded."),
    dict(_GB_REPORT, story_id="gb_injury_report_0921_terrell", entity_id="ATL", entity_type="team",
         team="ATL", game_id=GAME, category="def_absence", claim_key="terrell_status",
         claim_value="DNP (estimate)", claim="A.J. Terrell, CB (groin): Did Not Participate (estimated).",
         uncertainty="Estimated participation (Falcons walkthrough); may be superseded."),
    dict(story_id="atl_terrell_ir_headline", entity_id="ATL", entity_type="team", team="ATL",
         game_id=GAME, category="def_absence", claim_key="terrell_status", claim_value="injured reserve",
         claim="Headline: Falcons place A.J. Terrell Jr. on injured reserve.",
         attribution="Atlanta Falcons (related-content headline only)", source_tier="team_official",
         claim_kind="confirmed", source_url=_PENIX_URL, source_title="Related-content link on the team site",
         published_at=None, fetched_at="2026-09-23T01:39:07Z",
         rationale="The linked article was not fetched; its publication time is unknown."),
]


def role_records():
    base = dict(game_id=GAME, as_of=AS_OF, measurement_kind="observed", verified=True,
                support_scope="current_season", fetched_at="2026-09-22T19:00:00Z",
                reason_not_applied="no Week 3 run receipt; the projection's trailing usage window "
                                   "is not attributed to these two games here")
    return [fe.normalize_record(dict(base, factor_id="role:reed_snaps", category="role_usage",
                                     entity_id=PLAYER_IDS["Jayden Reed"], entity_type="player", team="GB",
                                     source_id="PFR snap counts via nflreadpy",
                                     observation="Jayden Reed offensive snaps 57% (Wk1) -> 3% (Wk2); neck, "
                                                 "estimated DNP on the Sept. 21 report", support_games=2)),
            fe.normalize_record(dict(base, factor_id="role:golden_targets", category="role_usage",
                                     entity_id=PLAYER_IDS["Matthew Golden"], entity_type="player", team="GB",
                                     source_id="nflverse play-by-play; PFR snap counts",
                                     observation="Matthew Golden 18 targets = 26.1% team share, "
                                                 "153 yds; offensive snaps 82% / 86%", support_games=2))]


def schedule_record():
    games = [{"game_id": "2026_03_ATL_GB", "week": 3, "kickoff": "2026-09-24T19:15:00-05:00",
              "network": "Prime Video"},
             {"game_id": "2026_04_ATL_NO", "week": 4, "kickoff": "2026-10-05T19:15:00-05:00",
              "network": "ESPN"},
             {"game_id": "2026_05_BAL_ATL", "week": 5, "kickoff": "2026-10-11T20:20:00-04:00",
              "network": "NBC"}]
    return fe.schedule_context_record("ATL", games, AS_OF, sources=[
        {"source_url": "https://www.atlantafalcons.com/schedule/", "source_tier": "team_official",
         "source_id": "atlantafalcons.com schedule (raw HTML)",
         "fetched_at": "2026-09-23T01:39:08Z", "agrees": True},
        {"source_url": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
         "source_tier": "data_feed", "fetched_at": "2026-09-22T18:25:11Z", "agrees": True}])


def split_records():
    love = [("2023_04_DET_GB", "2023-09-28", 246), ("2023_13_KC_GB", "2023-12-03", 267),
            ("2024_13_MIA_GB", "2024-11-28", 274), ("2024_16_NO_GB", "2024-12-23", 182),
            ("2025_02_WAS_GB", "2025-09-11", 292), ("2025_10_PHI_GB", "2025-11-10", 176)]
    return [
        fe.split_context_record(entity_id=PLAYER_IDS["Jordan Love"], entity_type="player", team="GB",
                                game_id=GAME, as_of=AS_OF, split_kind="primetime",
                                stat="passing yards (Lambeau x primetime, as primary passer)",
                                split_mean=239.5, split_n=6, baseline_mean=237.4, baseline_n=53,
                                cutoff="2026-09-21", source_id="nflverse play-by-play",
                                games=[{"game_id": g, "gameday": d, "value": v} for g, d, v in love]),
        fe.split_context_record(entity_id=PENIX, entity_type="player", team="ATL", game_id=GAME,
                                as_of=AS_OF, split_kind="venue", stat="passing yards at Lambeau",
                                split_mean=None, split_n=0, baseline_mean=226.6, baseline_n=12,
                                cutoff="2026-09-21", source_id="nflverse play-by-play", games=[]),
    ]


def weather_record():
    return fe.normalize_record(dict(
        factor_id="weather:2026_03_ATL_GB", category="weather", entity_id=GAME, entity_type="game",
        game_id=GAME, as_of=AS_OF, measurement_kind="projected",
        observation="NWS forecast for the 19:00 CDT kickoff hour: 60 F, wind 1 mph, 0% precipitation",
        source_url="https://api.weather.gov/gridpoints/GRB/78,31/forecast/hourly",
        source_id="NWS GRB/78,31", published_at="2026-09-22T17:36:16Z",
        fetched_at="2026-09-22T18:25:00Z", verified=True,
        rationale="Calm, dry and mild at kickoff.",
        uncertainty="Forecast issued two days before kickoff; refresh before use.",
        reason_not_applied="no weather multiplier on the projected mean in the deployed path; live runs "
                           "stamp forecast temp/wind into features read by the ordering model, and "
                           "whether that happened for Week 3 is not verified (no run receipt)"))


def model_stage_records():
    """No Week 3 run receipt: configuration alone is not evidence a stage ran."""
    reason = ("no Week 3 run receipt; code configuration names the {what} as shadow, but its "
              "execution is not verified")
    return [fe.normalize_record(dict(
        factor_id=f"{fid}:{GAME}", category=fid, entity_id=GAME, entity_type="game", game_id=GAME,
        as_of=AS_OF, measurement_kind="unavailable", populated=False, verified=False,
        source_id="code configuration (not a run receipt)", reason_not_applied=reason.format(what=what)))
        for fid, what in (("game_script", "score-based game-script margin"),
                          ("dispersion", "mean-conditional SD for passing yards"))]


def build_records():
    return (fe.assess_news(NEWS, AS_OF) + role_records() + [schedule_record(), weather_record()]
            + split_records() + model_stage_records())


def build_example_panel():
    return fe.build_panel(build_records(), AS_OF)


if __name__ == "__main__":
    print(json.dumps(build_example_panel(), indent=2))
