"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, stdlib only).

Forecasts every GTA VI PS5 trophy's Exophase earn rate at the case target.

Model, per trophy (my own design, not the house baseline):
  1. Trend: least-squares fit of logit(rate) against log(1 + hours since
     2026-11-19T00:00Z) over the case's last 24h of reads, extrapolated to
     the target. The slope is shrunk toward the tier's mean slope, and the
     extrapolated move from the last logit is damped (x0.8) so a 4.5-day
     extrapolation of a 1-day trend cannot run away.
  2. Earners/players decomposition as a cross-check: fit log(earners) and
     log(players tracked) against log(1 + hours) the same way; when the
     player count actually varied in history, blend the implied
     rate = earners_target / players_target (in logit space, 50/50) with
     the trend forecast. This handles dilution explicitly: a jump in
     players tracked mechanically lowers every rate, while earners keep
     growing as players progress.
  3. Flat-history dilution prior: when no read in the history ever changed
     a rate (launch night: every read identical, 638 players), the trend
     is flat but a refresh is overdue. Exophase's first refresh took the
     tracked base 638 -> 3,868 while every rate but the 100% one fell, so
     the median is scaled by 638/3868 and then back up by a small
     earnings factor (new players earn early trophies fast; platinum
     ~none). This branch only fires on a perfectly flat history, which
     the final case (two days of 10-minute reads) will not have.
  4. Spread: random-walk term 0.5*sqrt(days) plus the fit's residual
     spread plus a refresh-shock term (large when a refresh is overdue,
     smaller once refreshes are observed regularly). Saturated trophies
     (rate 100%) get a near-zero spread: everybody already has them.

Quantiles are sigmoid(median_logit + z_tau * sigma) * 100, clamped to
[0.01, 99.99], rounded to 4 decimals, non-decreasing by construction.

Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Reads every *.in in the input dir, writes <id>.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Launch-night refresh, documented in the kit rules: tracked players went
# 638 -> 3,868 at Exophase's first refresh, every rate but 100% falling.
DILUTION_RATIO = 638.0 / 3868.0
# Earnings factor applied after a dilution: new tracked players do earn
# early trophies quickly, so the rate does not fall by the full ratio.
EARN_FACTOR = {"platinum": 1.00, "gold": 1.05, "silver": 1.10, "bronze": 1.15}
# Fallback median rate (%) when a trophy has no usable history at all.
TIER_PRIOR = {"platinum": 0.5, "gold": 1.0, "silver": 0.8, "bronze": 1.0}
DAMP = 0.8          # damping on the extrapolated logit move
MIN_SIGMA = 0.35    # floor on the logit spread: never claim certainty


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(at: datetime) -> float:
    return max(0.0, (at - RELEASE).total_seconds() / 3600.0)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = 0.0
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    return (a, b, resid)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    # --- history in (x, players, earners) form, oldest first ---------------
    reads = []
    for h in history:
        t = parse_time(h["at"])
        reads.append((math.log1p(hours_since_release(t)), t, h["players_tracked"], h["rates"]))
    last_x, _, last_players, last_rates = reads[-1]
    last_time = reads[-1][1]
    recent = [r for r in reads if (last_time - r[1]).total_seconds() <= 24 * 3600]
    if len(recent) < 3:
        recent = reads

    # Player-count behaviour: did it ever move, and how fast does it grow?
    players_varied = len({r[2] for r in reads}) > 1
    change_gaps = []
    prev_p = None
    prev_t = None
    for _, t, p, _ in reads:
        if prev_p is not None and p != prev_p and prev_t is not None:
            change_gaps.append((t - prev_t).total_seconds() / 3600.0)
        prev_p, prev_t = p, t
    cadence_h = sorted(change_gaps)[len(change_gaps) // 2] if change_gaps else 6.0
    cadence_h = min(max(cadence_h, 1.0), 24.0)
    logp_fit = ols([r[0] for r in recent], [math.log(max(1, r[2])) for r in recent]) if players_varied else None

    n_trophies = len(case["trophies"])
    # Tier mean slopes, for shrinkage (computed from the trend fits below).
    slopes = {}
    trend = {}
    for i, trophy in enumerate(case["trophies"]):
        pts = [(r[0], logit(r[3][i])) for r in recent
               if i < len(r[3]) and r[3][i] is not None]
        fit = ols([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 2 else None
        trend[i] = (fit, pts)
        if fit is not None:
            slopes.setdefault(trophy["tier"], []).append(fit[1])
    tier_slope = {t: (sum(v) / len(v) if v else 0.0) for t, v in slopes.items()}

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        tier = trophy["tier"]
        last_rate = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else None
        if last_rate is None:
            # Fall back to the last non-null read for this trophy.
            for r in reversed(reads):
                if i < len(r[3]) and r[3][i] is not None:
                    last_rate = r[3][i]
                    break
        if last_rate is None:
            last_rate = TIER_PRIOR.get(tier, 1.0)
        last_lg = logit(last_rate)

        # Saturated trophy: everybody has it, it stays at 100%.
        if last_rate >= 99.99:
            q = [round4(clamp(sigmoid(logit(99.99) + z * 0.05))) for z in Z]
            rows.append({"id": trophy["id"], "q": q})
            continue

        fit, pts = trend[i]
        distinct = len({round(y, 6) for _, y in pts})
        if fit is None or distinct < 2:
            # Perfectly flat history: dilution prior (see module docstring).
            med_lg = logit(last_rate * DILUTION_RATIO * EARN_FACTOR.get(tier, 1.1))
            sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2 + 0.4 ** 2 + 0.2 ** 2)
            q = [round4(clamp(sigmoid(med_lg + z * sigma))) for z in Z]
            rows.append({"id": trophy["id"], "q": q})
            continue

        a, b, resid = fit
        # Shrink the slope toward the tier mean (weight grows with points).
        w = len(pts) / (len(pts) + 4.0)
        b_shrunk = w * b + (1.0 - w) * tier_slope.get(tier, 0.0)
        # Extrapolate, damped, from the fitted value at the last read.
        fit_last = a + b * last_x
        med_lg = fit_last + DAMP * b_shrunk * (x_target - last_x)
        # Blend with the earners/players decomposition when players varied.
        if players_varied and logp_fit is not None:
            e_pts = [(r[0], math.log(max(1.0, (r[3][i] or 0.0) / 100.0 * r[2] + 1.0)))
                     for r in recent if i < len(r[3]) and r[3][i] is not None]
            e_fit = ols([p[0] for p in e_pts], [p[1] for p in e_pts]) if len(e_pts) >= 3 else None
            if e_fit is not None:
                e_t = math.exp(e_fit[0] + e_fit[1] * x_target)
                p_t = math.exp(logp_fit[0] + logp_fit[1] * x_target)
                decomp_lg = logit(100.0 * e_t / max(1.0, p_t))
                med_lg = 0.5 * med_lg + 0.5 * decomp_lg

        # Spread: random walk + fit residual + refresh shock.
        expected_refreshes = horizon_hours / cadence_h
        refresh_sigma = min(0.8, 0.35 * math.sqrt(expected_refreshes))
        sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2
                          + max(resid, 0.15) ** 2 + refresh_sigma ** 2)
        sigma = max(sigma, MIN_SIGMA)
        q = [round4(clamp(sigmoid(med_lg + z * sigma))) for z in Z]
        # Enforce non-decreasing after rounding.
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
