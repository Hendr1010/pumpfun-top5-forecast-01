// pumpfun-top5-forecast-01 v3: trajectory-extrapolation forecast.
// v3 changes vs v2 (round-2 entry): velocity capped at 0.25/step (one-step
// spikes do not persist), velocity decay tau 3 -> 2 steps, unnamed share base
// 0.45 -> 0.60 (newcomers take most top-5 places over a 7-13h horizon), and
// Monte Carlo sd 0.15/step -> 0.04/step with base 0 (the ranking sd is the
// idiosyncratic volatility; common market moves do not change ranks).
// Last-read order score on the practice: -19.78 vs v2 -21.27, baseline -23.20.
// Reads /input/task.json + /input/history.ndjson, writes /output/forecast.json.
// Node 22, standard library only, deterministic (seeded RNG), no network.
//
// Model (beats the persistence baseline by naming the coins that WILL be top 5,
// not the ones that ARE):
//  - Per coin: last log-cap, recent per-step velocity (recency-weighted), age,
//    venue (pool/curve), number of observations.
//  - Project each coin's log-cap to every answer read with an exponentially
//    decaying velocity (launch spike fades, tau = 3 steps) plus a mild drift.
//  - Monte Carlo (seeded) over the 7 reads: sample each coin's projected
//    log-cap from its uncertainty, rank, record place holders. Average over
//    reads and samples -> P(coin at place k | a history coin holds it).
//  - Probabilities: p = (1 - P_NEW) * q for the top coins per place; the rest
//    is left unnamed for newcomers / coins too small to name. P_NEW = 0.50.
//  - Sizes: per place, a normal in log10(cap in SOL) centred on the projected
//    cap of the k-th ranked coin at the median read, sd widening with horizon.
//  - top5: the top 5 by projected log-cap at the median read.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;

// ---- tunables (fixed in advance, from the practice analysis, not tuned per run) ----
const TAU_STEPS = 2; // velocity decay timescale, in 10-min steps (v3: faster fade)
const DRIFT_PER_STEP = -0.01; // slow mean-reversion drift per step
const VEL_CAP = 0.25; // cap |velocity| per step (v3: shrink one-step spikes)
// Share of each place left for unnameable coins (newcomers). Grows with the
// mean horizon: over 1h ~half the top 5 turns over to coins the history cannot
// name; over 10h (the final) most of it does. Capped at 0.65.
const P_NEW_BASE = 0.60; // (v3: newcomers take most top-5 places at long horizon)
const P_NEW_PER_HOUR = 0.015;
const P_NEW_CAP = 0.65;
const MAX_NAMED_PER_PLACE = 12; // coins named per place
const MC_SAMPLES = 1500; // Monte Carlo samples per read
const SD0 = 0; // base log-cap uncertainty (v3: current cap is observed, not uncertain)
const SIGMA_STEP = 0.04; // per-step volatility added in quadrature (v3: idiosyncratic only;
// common market moves do not change ranks, so the ranking sd is far below total volatility)
const SIZE_SD0 = 0.35; // base size sd (as the baseline)
const SIZE_SD_PER_HOUR = 0.05; // size sd growth per hour of horizon

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
// prior velocity for coins with a single observation: median velocity of coins
// that newly entered the top 20 at the last mark is positive; use +0.10.
const PRIOR_VEL = 0.1;
for (const c of coins.values()) {
  const o = c.obs;
  const last = o[o.length - 1]!;
  // velocity from the last up to 3 steps, recency-weighted (weights 4,2,1)
  let num = 0;
  let den = 0;
  const steps = Math.min(3, o.length - 1);
  for (let s = 1; s <= steps; s++) {
    const a = o[o.length - 1 - s]!;
    const b = o[o.length - s]!;
    const dtSteps = Math.max(1, Math.round((b.at - a.at) / cadenceMs));
    const v = (b.log - a.log) / dtSteps;
    const w = Math.pow(2, steps - s); // most recent step weighs most
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

// ---- Monte Carlo over the 7 reads ----
// q[place][mint] = P(coin at place | a history coin holds it), averaged over reads.
const q: Array<Map<string, number>> = [];
for (let k = 0; k < places; k++) q.push(new Map());
const n = feats.length;
// Precompute per-read projected means and sds.
const readProj: Array<Array<{ mint: string; mean: number; sd: number }>> = reads.map((t) => {
  const sd = projectedSd(t);
  return feats.map((f) => ({ mint: f.mint, mean: projectedLog(f, t), sd }));
});
for (let ri = 0; ri < reads.length; ri++) {
  const proj = readProj[ri]!;
  const counts: Array<Map<string, number>> = [];
  for (let k = 0; k < places; k++) counts.push(new Map());
  // Sort once by mean to find the plausible top set; sample only the top 60 by
  // mean plus a random tail, to keep it fast. Ranking is dominated by the top.
  const order = [...proj].sort((a, b) => b.mean - a.mean);
  const pool = order.slice(0, 60);
  for (let s = 0; s < MC_SAMPLES; s++) {
    const sampled = pool.map((p) => ({ mint: p.mint, v: p.mean + p.sd * randn() }));
    sampled.sort((a, b) => b.v - a.v);
    for (let k = 0; k < places; k++) {
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
const sizeSd = Math.sqrt(SIZE_SD0 * SIZE_SD0 + Math.pow(SIZE_SD_PER_HOUR * meanHorizonH, 2));
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
const above = (x: number, mean: number, sd: number): number =>
  x === -Infinity ? 1 : x === Infinity ? 0 : 0.5 * erfc((x - mean) / (sd * Math.SQRT2));
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
