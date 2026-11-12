# pumpfun-top5-forecast-01 entry

Predict the top 5 new Solana pump.fun coins by market cap, 24 hours out.

`main.ts` (Node 22, standard library only, deterministic, no network) reads
`/input/task.json` + `/input/history.ndjson` and writes `/output/forecast.json`.

## Model

The forecast names the coins that **will** be top 5, not the ones that are:

1. **Per-coin trajectory.** Last log-cap, recency-weighted per-step velocity
   (last ≤3 steps, weights 4/2/1), age, venue, observation count.
   Single-observation coins take a prior velocity (+0.10/step).
2. **Projection.** Each coin's log-cap is projected to every answer read
   (times read from `task.json`): velocity decays exponentially
   (`tau = 3` steps) plus a mild downward drift (`-0.01`/step), so the long
   final horizon (7–13h) cannot run away.
3. **Seeded Monte Carlo** (mulberry32 + Box-Muller, 1500 samples/read, top-60
   pool by projected mean): sample each coin's projected log-cap, rank,
   record place holders. Average over the 7 reads → q(coin, place).
4. **Probabilities.** `p = (1 - P_NEW) * q` for the top 12 coins per place.
   `P_NEW = min(0.65, 0.45 + 0.015 * meanHorizonHours)` — the share left
   unnamed for newcomers (coins born after the history ends can take places,
   and no forecast can name them).
5. **Sizes.** Per place, a normal in log10(cap in SOL) centred on the
   projected cap of the k-th ranked coin at the median read, sd widening
   with the mean horizon, discretised into the 82 size bins.
6. **top5.** Top 5 by projected log-cap at the median read.

## Practice evidence

On the kit's 2-hour rehearsal (6 marks, 20:00–20:50Z): only 27 of 454
coins appear in both the first and last mark; top-5 churn is extreme early
then stabilises. Velocity predicts displacement — MIHAO VI (+0.25/step at
#7) ended #2 at the last answer read. 3 of the 5 last-read top coins were
not in the history rows, so a large unnamed share is essential. On the
only answer read whose top 5 is published, this model's order score is
−21.27 vs the house baseline's −23.20 (+1.93 better).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Runs in ~0.5s on the practice input; two runs are byte-identical.
