// pumpfun-top5-forecast-01 entry: trajectory-extrapolation forecast.
// Reads /input/task.json + /input/history.ndjson, writes /output/forecast.json.
// Node 22, standard library only, deterministic (seeded RNG), no network.
//
// The forecast names the coins that WILL be top 5, not the ones that ARE:
//  - Per coin: last log-cap, recency-weighted per-step velocity, age, venue,
//    number of observations. Single-observation coins take a prior velocity.
//  - Project each coin's log-cap to every answer read (times read from
//    task.json): velocity decays exponentially (launch spike fades, tau = 3
//    steps) plus a mild downward drift, so the long final horizon (7-13h)
//    cannot run away.
//  - Seeded Monte Carlo over the 7 reads: sample each coin's projected
//    log-cap from its uncertainty, rank, record place holders. Average over
//    reads and samples -> P(coin at place k | a history coin holds it).
//  - Probabilities: p = (1 - P_NEW) * q for the top coins per place; the rest
//    stays unnamed for newcomers / coins too small to name. P_NEW grows with
//    the mean horizon (over 1h ~half the top 5 turns over to coins the history
//    cannot name; over 10h most of it does).
//  - Sizes: per place, a normal in log10(cap in SOL) centred on the projected
//    cap of the k-th ranked coin at the median read, sd widening with horizon.
//  - top5: the top 5 by projected log-cap at the median read.
//
// Practice evidence (6 marks, 20:00-20:50Z): only 27 of 454 coins appear in
// both the first and last mark; top-5 churn is extreme early then stabilises.
// Velocity predicts displacement (MIHAO VI +0.25/step at #7 ended #2 at the
// last answer read). 3 of the 5 last-read top coins were NOT in the history
// rows, so a large unnamed share is essential. Pool (graduated) coins are
// durable; curve coins decay on average.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;

// ---- tunables (fixed in advance, from the practice analysis) ----
const TAU_STEPS = 3; // velocity decay timescale, in 10-min steps
const DRIFT_PER_STEP = -0.01; // slow mean-reversion drift per step
const VEL_CAP = 0.5; // cap |velocity| per step
// Share of each place left for unnameable coins (newcomers). Grows with the
// mean horizon; capped at 0.65.
const P_NEW_BASE = 0.45;
const P_NEW_PER_HOUR = 0.015;
const P_NEW_CAP = 0.65;
const MAX_NAMED_PER_PLACE = 12; // coins named per place
const MC_SAMPLES = 1500; // Monte Carlo samples per read
const MC_POOL = 60; // coins sampled per read (top by projected mean)
const SD0 = 0.12; // base log-cap uncertainty
const SIGMA_STEP = 0.15; // per-step volatility added in quadrature
const SIZE_SD0 = 0.35; // base size sd (as the house baseline)
const SIZE_SD_PER_HOUR = 0.05; // size sd growth per hour of horizon
const PRIOR_VEL = 0.1; // prior velocity for single-observation coins

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
}

interface Mark {
  ok: boolean;
  at: string;
  rows: Row[] | null;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  size: { edges: number[] };
  t0: string;
  history_until: string;
  reads: string[];
  cadence_minutes: number;
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const marks = lines
  .map((l) => JSON.parse(l) as Mark)
  .filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const places = task.places;
const edges = task.size.edges;
const reads = task.reads.map((s) => Date.parse(s));
const cadenceMs = (task.cadence_minutes || 10) * 60 * 1000;
const lastMarkAt = Date.parse(marks[marks.length - 1]!.at);

/** log10 of a cap in SOL, as the scorer takes it: log10(lamports) - 9. */
const logcap = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

// ---- per-coin trajectories ----
interface Coin {
  mint: string;
  obs: Array<{ at: number; log: number; rank: number; venue: string; createdAt: number }>;
}
const coins = new Map<string, Coin>();
for (const m of marks) {
  const at = Date.parse(m.at);
  for (const r of m.rows!) {
    let c = coins.get(r.mint);
    if (!c) {
      c = { mint: r.mint, obs: [] };
      coins.set(r.mint, c);
    }
    c.obs.push({ at, log: logcap(r.market_cap_lamports), rank: r.rank, venue: r.venue, createdAt: Date.parse(r.created_at) });
  }
}
for (const c of coins.values()) c.obs.sort((a, b) => a.at - b.at);

// ---- per-coin features ----
interface Feat {
  mint: string;
  lastLog: number;
  lastRank: number;
  velocity: number; // per-step log-cap change, recency-weighted
  ageMin: number;
  venue: string;
  nObs: number;
}
const feats: Feat[] = [];
for (const c of coins.values()) {
  const o = c.obs;
  const last = o[o.length - 1]!;
  // Velocity from the last up to 3 steps, recency-weighted (weights 4, 2, 1:
  // the most recent step weighs most).
  let num = 0;
  let den = 0;
  const steps = Math.min(3, o.length - 1);
  for (let s = 1; s <= steps; s++) {
    const a = o[o.length - 1 - s]!;
    const b = o[o.length - s]!;
    const dtSteps = Math.max(1, Math.round((b.at - a.at) / cadenceMs));
    const v = (b.log - a.log) / dtSteps;
    const w = Math.pow(2, steps - s);
    num += w * v;
    den += w;
  }
  let velocity = den > 0 ? num / den : PRIOR_VEL;
  if (!Number.isFinite(velocity)) velocity = PRIOR_VEL;
  velocity = Math.max(-VEL_CAP, Math.min(VEL_CAP, velocity));
  feats.push({
    mint: c.mint,
    lastLog: last.log,
    lastRank: last.rank,
    velocity,
    ageMin: (last.at - last.createdAt) / 60000,
    venue: last.venue,
    nObs: o.length,
  });
}

// ---- seeded RNG (mulberry32) ----
let rngState = 0x9e3779b9;
function rng(): number {
  rngState |= 0;
  rngState = (rngState + 0x6d2b79f5) | 0;
  let t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
// Box-Muller normal sample (deterministic given rng()).
let spare: number | null = null;
function randn(): number {
  if (spare !== null) {
    const v = spare;
    spare = null;
    return v;
  }
  let u = 0;
  let v = 0;
  do {
    u = rng();
  } while (u === 0);
  v = rng();
  const r = Math.sqrt(-2 * Math.log(u));
  const th = 2 * Math.PI * v;
  spare = r * Math.sin(th);
  return r * Math.cos(th);
}

// ---- projection ----
// Projected log-cap of a coin at time t (ms). Velocity decays with tau; drift applies.
function projectedLog(f: Feat, t: number): number {
  const dtSteps = Math.max(0, (t - lastMarkAt) / cadenceMs);
  const boost = f.velocity * TAU_STEPS * (1 - Math.exp(-dtSteps / TAU_STEPS));
  return f.lastLog + boost + DRIFT_PER_STEP * dtSteps;
}
// Uncertainty (sd) of the projected log-cap at time t.
function projectedSd(t: number): number {
  const dtSteps = Math.max(0, (t - lastMarkAt) / cadenceMs);
  return Math.sqrt(SD0 * SD0 + SIGMA_STEP * SIGMA_STEP * dtSteps);
}

// ---- Monte Carlo over the reads ----
// q[place][mint] = P(coin at place | a history coin holds it), averaged over reads.
const q: Array<Map<string, number>> = [];
for (let k = 0; k < places; k++) q.push(new Map());
// Precompute per-read projected means and sds.
const readProj: Array<Array<{ mint: string; mean: number; sd: number }>> = reads.map((t) => {
  const sd = projectedSd(t);
  return feats.map((f) => ({ mint: f.mint, mean: projectedLog(f, t), sd }));
});
for (let ri = 0; ri < reads.length; ri++) {
  const proj = readProj[ri]!;
  const counts: Array<Map<string, number>> = [];
  for (let k = 0; k < places; k++) counts.push(new Map());
  // Sort once by mean to find the plausible top set; sample only the top 60
  // by mean, to keep it fast. Ranking is dominated by the top.
  const order = [...proj].sort((a, b) => b.mean - a.mean);
  const pool = order.slice(0, MC_POOL);
  // A history smaller than the number of places can only fill that many
  // places per sample; record only the places the pool actually holds.
  const placesHere = Math.min(places, pool.length);
  for (let s = 0; s < MC_SAMPLES; s++) {
    const sampled = pool.map((p) => ({ mint: p.mint, v: p.mean + p.sd * randn() }));
    sampled.sort((a, b) => b.v - a.v);
    for (let k = 0; k < placesHere; k++) {
      const m = sampled[k]!.mint;
      counts[k]!.set(m, (counts[k]!.get(m) ?? 0) + 1);
    }
  }
  for (let k = 0; k < places; k++) {
    for (const [mint, c] of counts[k]!) {
      q[k]!.set(mint, (q[k]!.get(mint) ?? 0) + c / MC_SAMPLES / reads.length);
    }
  }
}

// ---- probabilities ----
// Mean horizon in hours (last mark -> answer reads), for P_NEW and size sd.
const meanHorizonH = reads.reduce((s, t) => s + (t - lastMarkAt) / 3600000, 0) / reads.length;
const P_NEW = Math.min(P_NEW_CAP, P_NEW_BASE + P_NEW_PER_HOUR * meanHorizonH);
// Name the top coins per place; p = (1 - P_NEW) * q. The rest stays unnamed.
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  const entries = [...q[k]!.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_NAMED_PER_PLACE);
  for (const [mint, qq] of entries) {
    const p = Math.floor((1 - P_NEW) * qq * MICRO) / MICRO;
    if (p > 0) probabilities.push({ mint, place: k + 1, p });
  }
}

// ---- top5: projected ranking at the median read ----
const medT = reads[Math.floor(reads.length / 2)]!;
const medProj = feats
  .map((f) => ({ mint: f.mint, mean: projectedLog(f, medT) }))
  .sort((a, b) => b.mean - a.mean);
const top5 = medProj.slice(0, places).map((p) => p.mint);

// ---- sizes ----
// Per place, a normal in log10(cap in SOL) centred on the projected cap of the
// k-th ranked coin at the median read, sd widening with the mean horizon.
/** erfc(x), Numerical Recipes' erfcc (fractional error under 1.2e-7 everywhere). */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z - 1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    );
  return x >= 0 ? r : 2 - r;
}
/** P(X >= x) for X normal with mean `mean` and sd `sd`. */
const above = (x: number, mean: number, sd: number): number =>
  x === -Infinity ? 1 : x === Infinity ? 0 : 0.5 * erfc((x - mean) / (sd * Math.SQRT2));
const sizeSd = Math.sqrt(SIZE_SD0 * SIZE_SD0 + Math.pow(SIZE_SD_PER_HOUR * meanHorizonH, 2));
const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  const mean = medProj[k]!.mean;
  for (let bin = 0; bin <= edges.length; bin++) {
    const lo = bin === 0 ? -Infinity : edges[bin - 1]!;
    const hi = bin === edges.length ? Infinity : edges[bin]!;
    const u = Math.floor((above(lo, mean, sizeSd) - above(hi, mean, sizeSd)) * MICRO);
    if (u > 0) sizes.push({ place: k + 1, bin, p: u / MICRO });
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
