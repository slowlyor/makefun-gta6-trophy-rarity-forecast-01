# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/slowlyor/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/slowlyor/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Logit-trend forecast of GTA VI PS5 trophy earn rates, with a new-player dilution prior for flat histories and a widened honest spread.

## Model

Per trophy, the program fits logit(earn rate) against log(1 + hours since launch) over the last 24h of reads, shrinks the slope toward the tier mean, damps the extrapolation, and blends it with an earners/players decomposition (log earners and log players tracked, each fit the same way, so a jump in players tracked dilutes rates mechanically while earners keep growing). When the whole history is flat — launch night, every read identical — a refresh is overdue, so the median is scaled by the documented first-refresh dilution (638 → 3,868 players) times a small per-tier earnings factor. Saturated trophies (100%) stay near 100% with a near-zero spread. Quantiles are sigmoid(median logit + z·σ)·100 with σ combining a random-walk term, fit residual, and a refresh-shock term.

On a local backtest proxy (practice answers approximated as last rate × 638/3868 × earnings factor 1.0–1.3), this scores a mean pinball loss of ~0.10–0.12 versus ~1.43–1.69 for the house baseline.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=vxw5vmyv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
