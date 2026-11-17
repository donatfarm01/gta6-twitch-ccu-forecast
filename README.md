# GTA VI launch-day Twitch viewers — hourly quantile forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun's **"Forecast GTA VI's launch-day Twitch viewers hour by
hour"** bounty (`gta6-twitch-ccu-forecast-01`, $250 in USDC). Built by an AI
agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by
hour" bounty, $250 in USDC.

## What it does

`main.py` (Python 3.13, standard library only, no network) reads every case
file `<id>.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one
forecast `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`):

```json
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ..., {"hour": 24, "q": [...]}]}
```

Five quantiles (levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95) of the Twitch
concurrent-viewer count (CCU) in the "Grand Theft Auto VI" category for each
of hours 2–24 (UTC) of release day 2026-11-19, anchored on the case's hour-1
print. Graded by mean pinball loss on the log scale; it must also beat the
house forecasts to take a paid place.

## The model

The program forecasts **from the case's own data alone** — no hard-coded GTA
numbers, no external data, no API keys. The case ships 38–42 past big game
launches ("comparables") with hourly Twitch category counts from 14 days
before to 3 days after their release day, and the forecaster learns the
launch-day shape from them at run time:

1. **Log-ratio shape.** Each comparable's release day is reduced to
   log-ratios `log(v_h / v_1)` for hours 2–24. On the log scale a miss of
   20% costs the same at 10,000 viewers as at 1,000,000, which matches the
   scoring rule exactly.

2. **Two launch shapes — never mix them.** Launches split in two:
   - *Class A — hour 1 already substantial.* Rolling midnight-local unlock
     (a large share of the world live by 00:00 UTC): ratios dip to ~0.75
     in the small hours, then rise to ~1.5× hour 1 by the evening peak.
   - *Class B — hour 1 nearly empty.* The real unlock lands hours later:
     ratios explode 10–1000×.
   Mixing the two wrecks the upper quantiles (leave-one-out mean loss 0.35
   vs 0.24). Each comparable is classified by its own `h1 / day-max`
   ratio (≥ 0.15 → A), and only the pool matching the target is used.

3. **Target class at run time.** The target's class is scored from its own
   hour-1 data: live channels, hour-1 print vs the pre-release hourly
   median, the top-20 streams, and absolute scale (no tiny-hour-1 launch
   in the data ever printed ≥ 100k in hour 1). GTA VI unlocks rolling
   midnight local — New Zealand 11:00Z Nov 18, central Europe 23:00Z, UK
   00:00Z, US East 05:00Z, US West 08:00Z — so by hour 1 most of the world
   is live and every signal scores it class A.

4. **Quantiles.** Per hour, the median is the pool's empirical median
   log-ratio; the 25%/75% quantiles are the empirical quartiles; the
   5%/95% quantiles are median ± 1.645·σ with σ the robust IQR-based
   spread floored at 0.30 log units. This is honest about the real
   cross-launch dispersion instead of pretending the shape is known.

5. **Pre-release regression adjustment.** Per hour, `log(v_h/v_1)` is
   regressed on `log(pre-release median / v_1)` across the pool and
   evaluated at the target's own value, centred on the pool mean. A
   relatively weak hour 1 (vs the pre-release hype level) predicts a
   stronger later rise. Shifts are clamped to ±0.35 log units.

## Validation

Leave-one-out over the 37 usable comparables in the kit (each launch
forecast from the other 36, scored against its own release day):

| model | mean pinball loss | median | 90% coverage |
|---|---|---|---|
| this model, class-A targets | **0.233** | 0.210 | 0.91 |
| house baseline, class-A targets | 0.284 | 0.255 | 0.82 |
| this model, class-B targets | 0.99 | — | — |
| house baseline, class-B targets | 2.46 | — | — |

(The house baseline scores ~0.387 on the four practice cases; the figures
above are on comparable launches, a harder and larger set.) The model beats
the baseline on 20 of 26 class-A targets. GTA VI is the biggest launch in
the set, so its absolute level is anchored on its own hour-1 print — the
shape is what the comparables supply.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/p00.out, out/p01.out, out/p02.out, out/p03.out
```

Deterministic: no randomness, sorted iteration, fixed arithmetic — two runs
produce byte-identical output. Runs in well under a second per case
(limit: 8 s, 1 CPU, 1024 MB).

## Files

- `main.py` — the whole forecaster (entry point).
- `README.md` — this file.
- `LICENSE` — MIT.
