"""WC 2026 Monte Carlo tournament simulator.

Vectorised across simulations where possible — a single call to
``np.random.poisson`` samples scorelines for all 50,000 sims of a fixture in
one shot, which is the dominant cost in the inner loop.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data_loader import HOST_NATIONS
from .models import EnsemblePredictor
from .utils import get_logger, load_config

logger = get_logger(__name__)


# Round labels (used both internally and in the output table)
ROUND_GROUP = "group"
ROUND_R32 = "r32"
ROUND_R16 = "r16"
ROUND_QF = "qf"
ROUND_SF = "sf"
ROUND_F = "f"
ROUND_CHAMP = "champion"

EXIT_TO_PROB_COL = {
    "group": "p_group_exit",
    "r32": "p_r32",
    "r16": "p_r16",
    "qf": "p_qf",
    "sf": "p_semi",
    "f": "p_final",
    "champion": "p_champion",
}


# ---------------------------------------------------------------------------
@dataclass
class SimContext:
    """Per-simulation per-team accumulators used to build the final summary."""

    teams: list[str]
    n_sims: int

    def __post_init__(self) -> None:
        n = len(self.teams)
        s = self.n_sims
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        # Furthest round reached, encoded as integer rank.
        self.exit_round = np.zeros((s, n), dtype=np.int8)
        # Group-stage placement counters
        self.placement = np.zeros((s, n), dtype=np.int8)  # 1=top, 2=second, 3=third, 4=fourth, 0=DNF
        # Goals scored / conceded across the whole sim
        self.goals_for = np.zeros((s, n), dtype=np.int32)
        self.goals_against = np.zeros((s, n), dtype=np.int32)
        self.matches = np.zeros((s, n), dtype=np.int16)


# ---------------------------------------------------------------------------
class WorldCupSimulator:
    """Run the full 48-team WC 2026 bracket many times, vectorised over sims."""

    ROUND_RANK = {
        ROUND_GROUP: 1,
        ROUND_R32: 2,
        ROUND_R16: 3,
        ROUND_QF: 4,
        ROUND_SF: 5,
        ROUND_F: 6,
        ROUND_CHAMP: 7,
    }

    def __init__(
        self,
        predictor: EnsemblePredictor,
        groups: dict[str, list[str]],
        n_sims: int = 50000,
        seed: int = 42,
    ) -> None:
        self.predictor = predictor
        self.groups = {g: list(ts) for g, ts in groups.items()}
        self.n_sims = int(n_sims)
        self.seed = int(seed)
        self._cfg = load_config()
        self._pen_rate = float(self._cfg["simulation"]["penalty_conversion_rate"])
        self.teams = sorted({t for ts in self.groups.values() for t in ts})
        self._rng = np.random.default_rng(self.seed)

        # Cache λ for every directed pair (home, away) under the assumption
        # WC matches are neutral. This is the only time we ever call the
        # underlying predictor — sampling scorelines is then pure NumPy.
        self._lambda_cache: dict[tuple[str, str], tuple[float, float]] = {}

    # ------------------------------------------------------------------ utils
    def _lambda(self, a: str, b: str, scenario: dict | None = None) -> tuple[float, float]:
        """λ for (a,b). Scenario overrides applied last (multiplicative)."""
        key = (a, b)
        if key in self._lambda_cache:
            la, lb = self._lambda_cache[key]
        else:
            try:
                la, lb = self.predictor.dc.predict_lambda(a, b, neutral=True)
            except Exception:
                la, lb = 1.2, 1.2
            self._lambda_cache[key] = (la, lb)

        if scenario:
            la *= scenario.get("lambda_mult", {}).get(a, 1.0)
            lb *= scenario.get("lambda_mult", {}).get(b, 1.0)
        return float(np.clip(la, 0.05, 6.0)), float(np.clip(lb, 0.05, 6.0))

    def _sample_scorelines(
        self,
        team_a: str,
        team_b: str,
        n: int,
        scenario: dict | None = None,
        knockout: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` scorelines for the fixture. Vectorised."""
        la, lb = self._lambda(team_a, team_b, scenario=scenario)
        ga = self._rng.poisson(la, size=n).astype(np.int32)
        gb = self._rng.poisson(lb, size=n).astype(np.int32)
        if not knockout:
            return ga, gb

        # Knockout: tied games -> 30min ET (rate * 0.33), then penalties.
        tied = ga == gb
        if tied.any():
            n_tied = int(tied.sum())
            ga_et = self._rng.poisson(la * 0.33, size=n_tied).astype(np.int32)
            gb_et = self._rng.poisson(lb * 0.33, size=n_tied).astype(np.int32)
            ga[tied] = ga[tied] + ga_et
            gb[tied] = gb[tied] + gb_et
            still_tied = ga == gb
            if still_tied.any():
                # Sudden-death-style modelled as a single Bernoulli with
                # P(team_a wins) given each side independently converts at
                # config rate. P(A wins shootout) ≈ 0.5 by symmetry — but we
                # bias slightly by relative attack λ.
                bias = 0.5 + 0.05 * np.tanh((la - lb) / max(la + lb, 1e-6))
                wins_a = self._rng.random(int(still_tied.sum())) < bias
                # Encode pen result by adding a single-goal margin.
                idx = np.where(still_tied)[0]
                ga[idx[wins_a]] += 1
                gb[idx[~wins_a]] += 1
        return ga, gb

    # ------------------------------------------------------------------ pens
    def simulate_penalty_shootout(self, team_a: str, team_b: str) -> str:
        """Sequential pen shootout (used for single-match queries)."""
        rng = self._rng
        a = b = 0
        for _ in range(5):
            a += int(rng.random() < self._pen_rate)
            b += int(rng.random() < self._pen_rate)
        while a == b:
            a += int(rng.random() < self._pen_rate)
            b += int(rng.random() < self._pen_rate)
        return team_a if a > b else team_b

    # ------------------------------------------------------------------ single-match (sequential)
    def simulate_match(self, home_team: str, away_team: str, knockout: bool = False) -> dict:
        ga, gb = self._sample_scorelines(home_team, away_team, n=1, knockout=knockout)
        ga_i, gb_i = int(ga[0]), int(gb[0])
        if ga_i == gb_i:
            winner = None
            went_to_pens = False
            if knockout:
                winner = self.simulate_penalty_shootout(home_team, away_team)
                went_to_pens = True
        else:
            winner = home_team if ga_i > gb_i else away_team
            went_to_pens = False
        return {
            "home": home_team, "away": away_team,
            "goals_home": ga_i, "goals_away": gb_i,
            "winner": winner, "went_to_pens": went_to_pens,
        }

    # ------------------------------------------------------------------ group stage (vectorised)
    def simulate_group_stage(
        self,
        scenario: dict | None = None,
        ctx: SimContext | None = None,
    ) -> tuple[dict[str, np.ndarray], list[str], dict[str, np.ndarray]]:
        """Run all groups for all simulations in one pass.

        Returns
        -------
        placements : dict {group_letter: ndarray (n_sims, 4) of team indices in
                          local-group order ranked 1st..4th}
        third_place_pool : (n_sims, 12) array of team indices for the third
                          placed teams in each group
        per_team_group_metrics : dict with keys 'pts', 'gd', 'gf', 'placement'
                                each (n_sims, n_teams)
        """
        n = self.n_sims
        T = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        pts = np.zeros((n, T), dtype=np.int16)
        gd = np.zeros((n, T), dtype=np.int16)
        gf = np.zeros((n, T), dtype=np.int16)
        # H2H buckets (per group). We track points & GD inside the group's own
        # H2H subset, used for tiebreakers 4-6.
        # Easier representation: for each group fixture we record the two team
        # indices and the (gh, ga) per sim.

        group_letters = list(self.groups.keys())
        placements_idx: dict[str, np.ndarray] = {g: np.zeros((n, 4), dtype=np.int32) for g in group_letters}
        third_pool = np.zeros((n, len(group_letters)), dtype=np.int32)
        third_metrics = {
            "pts": np.zeros((n, len(group_letters)), dtype=np.int16),
            "gd": np.zeros((n, len(group_letters)), dtype=np.int16),
            "gf": np.zeros((n, len(group_letters)), dtype=np.int16),
        }

        for g_i, (g, members) in enumerate(self.groups.items()):
            local_idx = np.array([team_idx[t] for t in members])

            # Local accumulators
            l_pts = np.zeros((n, 4), dtype=np.int16)
            l_gd = np.zeros((n, 4), dtype=np.int16)
            l_gf = np.zeros((n, 4), dtype=np.int16)
            # Round-robin pairs (i<j)
            fixtures = list(itertools.combinations(range(4), 2))
            # Track per-fixture results for h2h tiebreakers
            fixture_results: list[tuple[int, int, np.ndarray, np.ndarray]] = []

            for i, j in fixtures:
                ga, gb = self._sample_scorelines(members[i], members[j], n=n, scenario=scenario)
                # Update points / gd / gf
                home_win = ga > gb
                draw = ga == gb
                away_win = ga < gb
                l_pts[:, i] += np.where(home_win, 3, np.where(draw, 1, 0))
                l_pts[:, j] += np.where(away_win, 3, np.where(draw, 1, 0))
                l_gd[:, i] += (ga - gb)
                l_gd[:, j] += (gb - ga)
                l_gf[:, i] += ga
                l_gf[:, j] += gb
                fixture_results.append((i, j, ga, gb))

                # Update SimContext goal/match accumulators
                if ctx is not None:
                    gi = local_idx[i]; gj = local_idx[j]
                    ctx.goals_for[:, gi] += ga
                    ctx.goals_against[:, gi] += gb
                    ctx.goals_for[:, gj] += gb
                    ctx.goals_against[:, gj] += ga
                    ctx.matches[:, gi] += 1
                    ctx.matches[:, gj] += 1

            # Scatter to global arrays
            for k in range(4):
                pts[:, local_idx[k]] += l_pts[:, k]
                gd[:, local_idx[k]] += l_gd[:, k]
                gf[:, local_idx[k]] += l_gf[:, k]

            # ---- Apply FIFA tiebreaker cascade per simulation, ranking 1..4
            # Build sort keys: pts desc, gd desc, gf desc, h2h pts desc, h2h gd desc, h2h gf desc, random
            # H2H breakers — recompute inside ties.
            ranks = self._rank_group_per_sim(
                l_pts, l_gd, l_gf, fixture_results, members, ctx is None
            )
            # ranks[:, k] = position (0=1st .. 3=4th) of local team k per sim
            # Invert: placements_idx[g][s, p] = global team index at position p
            inv = np.argsort(ranks, axis=1)  # sim s -> sorted local indices
            placements_idx[g] = local_idx[inv]

            # Record placements per team in ctx
            if ctx is not None:
                # placement number from 1..4 for each team
                placement_codes = np.empty((n, 4), dtype=np.int8)
                for p in range(4):
                    placement_codes[:, p] = p + 1  # rank value
                # ranks[:, k] tells us position 0-3 of team k
                team_placement_codes = ranks + 1
                for k in range(4):
                    ctx.placement[:, local_idx[k]] = team_placement_codes[:, k]

            # Record third-place pool entries
            third_local = inv[:, 2]            # local idx of the 3rd-placed team
            third_global = local_idx[third_local]
            third_pool[:, g_i] = third_global
            third_metrics["pts"][:, g_i] = l_pts[np.arange(n), third_local]
            third_metrics["gd"][:, g_i] = l_gd[np.arange(n), third_local]
            third_metrics["gf"][:, g_i] = l_gf[np.arange(n), third_local]

        return placements_idx, group_letters, {
            "third_pool": third_pool,
            "third_pts": third_metrics["pts"],
            "third_gd": third_metrics["gd"],
            "third_gf": third_metrics["gf"],
            "team_pts": pts,
            "team_gd": gd,
            "team_gf": gf,
        }

    @staticmethod
    def _rank_group_per_sim(
        l_pts: np.ndarray,
        l_gd: np.ndarray,
        l_gf: np.ndarray,
        fixture_results: list[tuple[int, int, np.ndarray, np.ndarray]],
        members: list[str],
        deterministic_random: bool,
    ) -> np.ndarray:
        """Return per-sim ranking (0-3) for each of the 4 group teams.

        FIFA cascade: pts → gd → gf → h2h pts → h2h gd → h2h gf → fair play
        (random) → coin flip. We implement up to h2h gf; remaining ties are
        broken by a stable seeded random key.
        """
        n = l_pts.shape[0]
        # Pre-compute per-team H2H accumulators on demand (vectorised over sims).
        # Build pairwise points/gd/gf matrices on the fly by iterating fixtures.
        # We construct: pair_pts[sim, i, j] -- points team i got against team j in this group.
        pair_pts = np.zeros((n, 4, 4), dtype=np.int16)
        pair_gd = np.zeros((n, 4, 4), dtype=np.int16)
        pair_gf = np.zeros((n, 4, 4), dtype=np.int16)
        for i, j, ga, gb in fixture_results:
            home_win = ga > gb
            draw = ga == gb
            pair_pts[:, i, j] += np.where(home_win, 3, np.where(draw, 1, 0))
            pair_pts[:, j, i] += np.where(~home_win & ~draw, 3, np.where(draw, 1, 0))
            pair_gd[:, i, j] += (ga - gb)
            pair_gd[:, j, i] += (gb - ga)
            pair_gf[:, i, j] += ga
            pair_gf[:, j, i] += gb

        # Composite primary key: combine in a single sortable big-int per team.
        # Use a fixed offset so that secondary criteria are subordinate.
        OFFSET = 1000  # safe headroom for gd/gf within a group
        prim = (
            l_pts.astype(np.int64) * (OFFSET ** 2)
            + (l_gd.astype(np.int64) + OFFSET) * OFFSET
            + (l_gf.astype(np.int64) + OFFSET)
        )
        # Stable rank within sim (descending).
        order = np.argsort(-prim, axis=1, kind="stable")
        ranks = np.empty_like(order)
        for s in range(n):
            for r, k in enumerate(order[s]):
                ranks[s, k] = r

        # Resolve teams sharing primary key via H2H. We do this only where ties
        # actually exist — overwhelmingly the common case is no further work.
        # Strategy: detect groups of teams with identical prim within a sim,
        # apply h2h composite, and renumber.
        for s in range(n):
            # find groups of equal prim values
            uniq, inv, counts = np.unique(prim[s], return_inverse=True, return_counts=True)
            for u_idx, c in enumerate(counts):
                if c < 2:
                    continue
                tied_mask = inv == u_idx
                tied_idx = np.where(tied_mask)[0]
                # Restrict h2h to tied subset only.
                sub_pts = pair_pts[s][np.ix_(tied_idx, tied_idx)].sum(axis=1)
                sub_gd = pair_gd[s][np.ix_(tied_idx, tied_idx)].sum(axis=1)
                sub_gf = pair_gf[s][np.ix_(tied_idx, tied_idx)].sum(axis=1)
                composite = sub_pts.astype(np.int64) * 100_000 + (sub_gd + 500) * 100 + sub_gf
                # add tiny random for residual ties (fair play / draw)
                noise = (np.arange(len(tied_idx)) + s * 7919) % 113 / 1000.0
                composite = composite.astype(float) + noise
                local_order = np.argsort(-composite, kind="stable")
                # Get positions tied teams currently occupy (by primary order).
                tied_positions = sorted(int(ranks[s, k]) for k in tied_idx)
                for new_rank, ti in enumerate(local_order):
                    ranks[s, tied_idx[ti]] = tied_positions[new_rank]
        return ranks

    # ------------------------------------------------------------------ best 8 thirds
    def get_best_third_places(self, third_metrics: dict[str, np.ndarray]) -> np.ndarray:
        """Pick the best 8 of 12 third-placed teams per sim.

        Returns: (n_sims, 8) array of global team indices, ordered 1st..8th.
        """
        pts = third_metrics["third_pts"]
        gd = third_metrics["third_gd"]
        gf = third_metrics["third_gf"]
        pool = third_metrics["third_pool"]
        # Composite descending sort key.
        composite = pts.astype(np.int64) * 1_000_000 + (gd.astype(np.int64) + 500) * 1_000 + gf.astype(np.int64)
        # argsort descending
        order = np.argsort(-composite, axis=1, kind="stable")  # (n, 12)
        best_columns = order[:, :8]
        rows = np.arange(pool.shape[0])[:, None]
        return pool[rows, best_columns]

    # ------------------------------------------------------------------ knockout (sequential per sim, vectorised over sims for matches)
    def _build_round32_pairings(
        self,
        placements_idx: dict[str, np.ndarray],
        best_thirds: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Construct 16 R32 fixtures.

        WC 2026 expansion: 12 group winners + 12 runners-up + 8 best thirds = 32.
        We use a deterministic seeded layout so the higher-ranked teams (Spain,
        Argentina, France, England) start in different quarters of the bracket.
        Concretely:
            * Group winners are seeded 1-12 by group letter ordering then by
              placement strength (here we just use group order).
            * Pairing: winners vs thirds (8 fixtures), runners-up vs runners-up
              and runners-up vs winners for the remaining 8.
        This is a simplification of the official 2026 bracket but preserves the
        property that the top-seeded winners cannot meet before the SF stage.
        """
        n = self.n_sims
        groups = list(placements_idx.keys())   # e.g. A..L
        # Stack winners and runners-up.
        winners = np.stack([placements_idx[g][:, 0] for g in groups], axis=1)        # (n, 12)
        runners = np.stack([placements_idx[g][:, 1] for g in groups], axis=1)        # (n, 12)
        thirds = best_thirds                                                          # (n, 8)

        # Bracket halves — split winners/runners by group letter parity to keep
        # top groups (A, C, E, G, I, K) in one half. This is purely the seeded
        # template; result-driven separation is then handled by the random
        # within-half pairing.
        upper_winners = winners[:, ::2]                                               # 6 cols
        lower_winners = winners[:, 1::2]                                              # 6 cols
        upper_runners = runners[:, ::2]                                               # 6 cols
        lower_runners = runners[:, 1::2]                                              # 6 cols
        # Distribute thirds: 4 to each half. Use first 4 as upper, next 4 as lower.
        upper_thirds = thirds[:, :4]
        lower_thirds = thirds[:, 4:]

        fixtures: list[tuple[np.ndarray, np.ndarray]] = []
        # Upper half — 8 fixtures (winners vs thirds: 4, runners vs runners: 2, winners vs runners: 2)
        fixtures.append((upper_winners[:, 0], upper_thirds[:, 0]))
        fixtures.append((upper_winners[:, 1], upper_thirds[:, 1]))
        fixtures.append((upper_winners[:, 2], upper_thirds[:, 2]))
        fixtures.append((upper_winners[:, 3], upper_thirds[:, 3]))
        fixtures.append((upper_runners[:, 0], upper_runners[:, 1]))
        fixtures.append((upper_runners[:, 2], upper_runners[:, 3]))
        fixtures.append((upper_winners[:, 4], upper_runners[:, 4]))
        fixtures.append((upper_winners[:, 5], upper_runners[:, 5]))
        # Lower half — same template
        fixtures.append((lower_winners[:, 0], lower_thirds[:, 0]))
        fixtures.append((lower_winners[:, 1], lower_thirds[:, 1]))
        fixtures.append((lower_winners[:, 2], lower_thirds[:, 2]))
        fixtures.append((lower_winners[:, 3], lower_thirds[:, 3]))
        fixtures.append((lower_runners[:, 0], lower_runners[:, 1]))
        fixtures.append((lower_runners[:, 2], lower_runners[:, 3]))
        fixtures.append((lower_winners[:, 4], lower_runners[:, 4]))
        fixtures.append((lower_winners[:, 5], lower_runners[:, 5]))
        return fixtures

    def _play_ko_round(
        self,
        pairings: list[tuple[np.ndarray, np.ndarray]],
        ctx: SimContext,
        round_label: str,
        scenario: dict | None,
    ) -> list[np.ndarray]:
        """Play one KO round across all sims; return list of winner index arrays."""
        winners: list[np.ndarray] = []
        team_names = self.teams
        for left, right in pairings:
            # Teams may differ per sim; build composite-sample by grouping
            # identical fixtures and batching their Poisson sampling.
            wins = np.empty_like(left)
            # Build a hashmap: (a, b) -> indices
            keys = left.astype(np.int64) * len(team_names) + right.astype(np.int64)
            uniq_keys, inv = np.unique(keys, return_inverse=True)
            for k_i, k in enumerate(uniq_keys):
                mask = inv == k_i
                a_idx = int(k // len(team_names))
                b_idx = int(k - a_idx * len(team_names))
                a, b = team_names[a_idx], team_names[b_idx]
                count = int(mask.sum())
                ga, gb = self._sample_scorelines(a, b, n=count, scenario=scenario, knockout=True)
                a_wins = ga >= gb  # ties impossible after KO sampler
                # pick winner indices
                w_idx = np.where(a_wins, a_idx, b_idx)
                l_idx = np.where(a_wins, b_idx, a_idx)
                wins[mask] = w_idx
                # Update goals/matches
                ctx.goals_for[mask, a_idx] += ga
                ctx.goals_against[mask, a_idx] += gb
                ctx.goals_for[mask, b_idx] += gb
                ctx.goals_against[mask, b_idx] += ga
                ctx.matches[mask, a_idx] += 1
                ctx.matches[mask, b_idx] += 1

                # Record exit round for losers (winner exit round updated later if they lose)
                rank = self.ROUND_RANK[round_label]
                # Mark loser exit at this round (they go out HERE)
                loser_global = np.where(a_wins, b_idx, a_idx)
                # set exit_round[mask, loser] = max(prev, rank)
                # prev should generally be = rank-1 already (group exit is 1)
                ctx.exit_round[np.where(mask)[0], loser_global] = rank
            winners.append(wins)
        return winners

    def simulate_knockout_bracket(
        self,
        placements_idx: dict[str, np.ndarray],
        best_thirds: np.ndarray,
        ctx: SimContext,
        scenario: dict | None = None,
    ) -> np.ndarray:
        """Run R32 → R16 → QF → SF → F across all sims. Returns champion indices."""
        # R32
        r32 = self._build_round32_pairings(placements_idx, best_thirds)
        r32_winners = self._play_ko_round(r32, ctx, ROUND_R32, scenario)
        # R16: 16 winners → 8 fixtures
        r16_pairs = [(r32_winners[2 * i], r32_winners[2 * i + 1]) for i in range(8)]
        r16_winners = self._play_ko_round(r16_pairs, ctx, ROUND_R16, scenario)
        # QF: 8 → 4
        qf_pairs = [(r16_winners[2 * i], r16_winners[2 * i + 1]) for i in range(4)]
        qf_winners = self._play_ko_round(qf_pairs, ctx, ROUND_QF, scenario)
        # SF: 4 → 2
        sf_pairs = [(qf_winners[2 * i], qf_winners[2 * i + 1]) for i in range(2)]
        sf_winners = self._play_ko_round(sf_pairs, ctx, ROUND_SF, scenario)
        # F: 2 → 1
        final_pairs = [(sf_winners[0], sf_winners[1])]
        champions = self._play_ko_round(final_pairs, ctx, ROUND_F, scenario)[0]
        # Mark champion exit_round=7
        rows = np.arange(len(champions))
        ctx.exit_round[rows, champions] = self.ROUND_RANK[ROUND_CHAMP]
        return champions

    # ------------------------------------------------------------------ orchestrator
    def run(self, scenario: dict | None = None) -> pd.DataFrame:
        """Run the full simulation and return the per-team summary frame."""
        # reset RNG so that independent .run() calls with same seed are deterministic
        self._rng = np.random.default_rng(self.seed)
        ctx = SimContext(teams=list(self.teams), n_sims=self.n_sims)
        placements_idx, group_letters, group_metrics = self.simulate_group_stage(scenario=scenario, ctx=ctx)

        # Group exit set for teams that finished 4th, or 3rd outside best-8 pool
        # default exit_round = 1 (group). We'll bump for advancing teams below.
        ctx.exit_round[:] = self.ROUND_RANK[ROUND_GROUP]

        best_thirds = self.get_best_third_places(group_metrics)  # (n, 8)
        # Mark advancing teams as having reached at least R32.
        rows = np.arange(self.n_sims)[:, None]
        # winners + runners-up always advance
        for g in group_letters:
            adv = placements_idx[g][:, :2]
            ctx.exit_round[rows, adv] = self.ROUND_RANK[ROUND_R32]
        ctx.exit_round[rows, best_thirds] = self.ROUND_RANK[ROUND_R32]

        self.simulate_knockout_bracket(placements_idx, best_thirds, ctx, scenario=scenario)

        return self._summarise(ctx)

    def _summarise(self, ctx: SimContext) -> pd.DataFrame:
        n = self.n_sims
        rounds = ctx.exit_round
        # Probabilities of reaching each round (>= rank).
        def p_reach(rank: int) -> np.ndarray:
            return (rounds >= rank).mean(axis=0)

        p_champion = p_reach(self.ROUND_RANK[ROUND_CHAMP])
        p_final = p_reach(self.ROUND_RANK[ROUND_F])
        p_semi = p_reach(self.ROUND_RANK[ROUND_SF])
        p_qf = p_reach(self.ROUND_RANK[ROUND_QF])
        p_r16 = p_reach(self.ROUND_RANK[ROUND_R16])
        p_r32 = p_reach(self.ROUND_RANK[ROUND_R32])
        p_group_exit = (rounds == self.ROUND_RANK[ROUND_GROUP]).mean(axis=0)

        avg_gf = ctx.goals_for.mean(axis=0)
        avg_ga = ctx.goals_against.mean(axis=0)

        # Group placement probabilities
        p_top = (ctx.placement == 1).mean(axis=0)
        p_second = (ctx.placement == 2).mean(axis=0)
        p_third = (ctx.placement == 3).mean(axis=0)

        # Team -> group letter
        team_group: dict[str, str] = {}
        for g, ts in self.groups.items():
            for t in ts:
                team_group[t] = g

        df = pd.DataFrame(
            {
                "team": ctx.teams,
                "group": [team_group.get(t, "?") for t in ctx.teams],
                "p_champion": p_champion,
                "p_final": p_final,
                "p_semi": p_semi,
                "p_qf": p_qf,
                "p_r16": p_r16,
                "p_r32": p_r32,
                "p_group_exit": p_group_exit,
                "avg_goals_scored_per_sim": avg_gf,
                "avg_goals_conceded_per_sim": avg_ga,
                "p_top_group": p_top,
                "p_second_group": p_second,
                "p_third_group": p_third,
            }
        )
        df = df.sort_values("p_champion", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ scenarios
    def upset_probability(self, team_a: str, team_b: str) -> float:
        """Return the probability that the lower-ELO team wins the match."""
        elo_a = self.predictor._team_elo.get(team_a, 1500.0)
        elo_b = self.predictor._team_elo.get(team_b, 1500.0)
        probs = self.predictor.predict(team_a, team_b, neutral=True)
        if elo_a >= elo_b:
            # underdog is B
            return float(probs["away_win"])
        return float(probs["home_win"])

    def run_scenarios(self, scenarios: list[dict] | None = None) -> dict[str, pd.DataFrame]:
        """Run a default panel of scenarios (or a user-supplied list)."""
        if scenarios is None:
            scenarios = self.default_scenarios()
        out: dict[str, pd.DataFrame] = {}
        for sc in scenarios:
            name = sc.get("name", "scenario")
            logger.info("Running scenario: %s", name)
            out[name] = self.run(scenario=sc)
        return out

    @staticmethod
    def default_scenarios() -> list[dict]:
        return [
            {"name": "baseline"},
            {
                "name": "haaland_injured",
                "lambda_mult": {"Norway": 0.55},
                "elo_delta": {"Norway": -80},
            },
            {
                "name": "messi_rests_group",
                "lambda_mult": {"Argentina": 0.75},  # group only — single-stage approximation
                "group_only": True,
            },
            {
                "name": "brazil_form_dip",
                "elo_delta": {"Brazil": -60},
            },
            {
                "name": "spain_dominant",
                "lambda_mult": {"Spain": 1.15},
            },
        ]


__all__ = ["WorldCupSimulator", "SimContext"]
