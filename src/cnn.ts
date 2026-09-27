// A tiny convolutional network with explicit gradients:
// 6×6 image → K filters of 3×3 (valid, stride 1) → activation → 2×2 pooling → dense → softmax.
import { random } from './engine';
import { argmax, createLab, softmax, type ParamModel, type Phase } from './lab';

export const IMAGE = 6, KERNEL = 3, MAP = IMAGE - KERNEL + 1, POOLED = MAP / 2, CLASSES = 3;
export type CnnProblem = 'lines' | 'shapes';
export type ConvActivation = 'relu' | 'tanh';
export type Pooling = 'max' | 'average';
export interface CnnConfig { filters: number; activation: ConvActivation; pooling: Pooling }
/** Params: `f{k}` holds filter k's 9 weights followed by its bias; `o{c}` holds class c's dense weights followed by its bias. */
export interface CnnModel extends ParamModel { config: CnnConfig }
export interface CnnImage { pixels: number[]; y: number }
export interface CnnForward { z: number[][]; a: number[][]; pooled: number[][]; winners: number[][]; flat: number[]; logits: number[]; probs: number[] }
export interface CnnTrace extends CnnForward {
  loss: number; target: number;
  dLogits: number[]; dFlat: number[]; dA: number[][]; dZ: number[][];
  grads: Record<string, number[]>;
}
export type CnnStage = 'input' | 'conv' | 'pool' | 'dense' | 'loss';
export interface CnnStep { phase: Phase; stage: CnnStage; k?: number; c?: number; update?: string[] }
export const DEFAULT_CNN: CnnConfig = { filters: 2, activation: 'relu', pooling: 'max' };

const SHAPES = [
  [0, 1, 0, 1, 1, 1, 0, 1, 0], // plus
  [1, 0, 1, 0, 1, 0, 1, 0, 1], // cross
  [1, 1, 1, 1, 0, 1, 1, 1, 1], // square outline
];
export function cnnDataset(problem: CnnProblem): CnnImage[] {
  const rng = random(problem === 'lines' ? 61 : 73);
  const noisy = (on: boolean) => on ? 0.8 + rng() * 0.2 : rng() * 0.18;
  const images: CnnImage[] = [];
  if (problem === 'shapes') {
    for (let n = 0; n < 16; n++) for (let y = 0; y < CLASSES; y++) {
      const top = Math.floor(((n * 5 + y * 3) % 16) / 4), left = (n * 5 + y * 3) % 4;
      const pixels = Array.from({ length: IMAGE * IMAGE }, (_, p) => {
        const r = Math.floor(p / IMAGE) - top, c = p % IMAGE - left;
        return noisy(r >= 0 && r < 3 && c >= 0 && c < 3 && SHAPES[y][r * 3 + c] === 1);
      });
      images.push({ pixels, y });
    }
    return images;
  }
  for (let n = 0; n < 8; n++) for (let y = 0; y < CLASSES; y++) {
    const row = Math.floor(rng() * IMAGE), start = Math.floor(rng() * 3), shift = Math.floor(rng() * 3), anti = rng() < 0.5;
    const pixels = Array.from({ length: IMAGE * IMAGE }, (_, p) => {
      const r = Math.floor(p / IMAGE), c = p % IMAGE;
      const on = y === 0 ? r === row && c >= start && c < start + 4
        : y === 1 ? c === row && r >= start && r < start + 4
        : r - shift >= 0 && r - shift < 4 && c - start >= 0 && c - start < 4 && (anti ? r - shift + c - start === 3 : r - shift === c - start);
      return noisy(on);
    });
    images.push({ pixels, y });
  }
  return images;
}

export function createCnn(config: CnnConfig, seed = 7): CnnModel {
  const rng = random(seed);
  const params: Record<string, number[]> = {};
  const filterScale = Math.sqrt(6 / (KERNEL * KERNEL + config.filters * 1));
  for (let k = 0; k < config.filters; k++) params[`f${k}`] = [...Array.from({ length: KERNEL * KERNEL }, () => (rng() * 2 - 1) * filterScale), 0];
  const inputs = config.filters * POOLED * POOLED;
  const denseScale = Math.sqrt(6 / (inputs + CLASSES));
  for (let c = 0; c < CLASSES; c++) params[`o${c}`] = [...Array.from({ length: inputs }, () => (rng() * 2 - 1) * denseScale), 0];
  return { config: { ...config }, params };
}
/** Indices in the 6×6 input covered by the 3×3 window at feature-map position (r, c). */
export function receptiveField(r: number, c: number) {
  return Array.from({ length: KERNEL * KERNEL }, (_, q) => (r + Math.floor(q / KERNEL)) * IMAGE + c + q % KERNEL);
}
/** Indices in a 4×4 feature map pooled into position (i, j) of the 2×2 output. */
export function poolWindow(i: number, j: number) {
  return [0, 1, 2, 3].map(q => (2 * i + Math.floor(q / 2)) * MAP + 2 * j + q % 2);
}
const act = (v: number, kind: ConvActivation) => kind === 'relu' ? Math.max(0, v) : Math.tanh(v);
const actDerivative = (z: number, a: number, kind: ConvActivation) => kind === 'relu' ? Number(z > 0) : 1 - a * a;

export function cnnForward(model: CnnModel, pixels: number[]): CnnForward {
  const { filters, activation, pooling } = model.config;
  const z: number[][] = [], a: number[][] = [], pooled: number[][] = [], winners: number[][] = [];
  for (let k = 0; k < filters; k++) {
    const f = model.params[`f${k}`];
    z[k] = Array.from({ length: MAP * MAP }, (_, p) => receptiveField(Math.floor(p / MAP), p % MAP).reduce((s, idx, q) => s + f[q] * pixels[idx], f[KERNEL * KERNEL]));
    a[k] = z[k].map(v => act(v, activation));
    winners[k] = []; pooled[k] = [];
    for (let i = 0; i < POOLED; i++) for (let j = 0; j < POOLED; j++) {
      const window = poolWindow(i, j);
      if (pooling === 'max') { const best = window.reduce((b, idx) => a[k][idx] > a[k][b] ? idx : b, window[0]); winners[k].push(best); pooled[k].push(a[k][best]); }
      else { winners[k].push(-1); pooled[k].push(window.reduce((s, idx) => s + a[k][idx], 0) / 4); }
    }
  }
  const flat = pooled.flat();
  const logits = Array.from({ length: CLASSES }, (_, c) => { const o = model.params[`o${c}`]; return flat.reduce((s, v, m) => s + o[m] * v, o[flat.length]); });
  return { z, a, pooled, winners, flat, logits, probs: softmax(logits) };
}
export function cnnTrace(model: CnnModel, sample: CnnImage): CnnTrace {
  const f = cnnForward(model, sample.pixels);
  const { filters, activation, pooling } = model.config;
  const loss = -Math.log(Math.max(f.probs[sample.y], 1e-300));
  // Softmax with cross-entropy: dL/dlogit = p − one-hot(target).
  const dLogits = f.probs.map((p, c) => p - Number(c === sample.y));
  const grads: Record<string, number[]> = {};
  for (let c = 0; c < CLASSES; c++) grads[`o${c}`] = [...f.flat.map(v => dLogits[c] * v), dLogits[c]];
  const dFlat = f.flat.map((_, m) => dLogits.reduce((s, d, c) => s + d * model.params[`o${c}`][m], 0));
  const dA: number[][] = [], dZ: number[][] = [];
  for (let k = 0; k < filters; k++) {
    dA[k] = Array(MAP * MAP).fill(0);
    for (let i = 0; i < POOLED; i++) for (let j = 0; j < POOLED; j++) {
      const g = dFlat[k * POOLED * POOLED + i * POOLED + j];
      // Max pooling routes the gradient to the winner only; average pooling shares it equally.
      if (pooling === 'max') dA[k][f.winners[k][i * POOLED + j]] += g;
      else for (const idx of poolWindow(i, j)) dA[k][idx] += g / 4;
    }
    dZ[k] = dA[k].map((g, p) => g * actDerivative(f.z[k][p], f.a[k][p], activation));
    const grad = Array(KERNEL * KERNEL + 1).fill(0);
    dZ[k].forEach((d, p) => {
      receptiveField(Math.floor(p / MAP), p % MAP).forEach((idx, q) => { grad[q] += d * sample.pixels[idx]; });
      grad[KERNEL * KERNEL] += d;
    });
    grads[`f${k}`] = grad;
  }
  return { ...f, loss, target: sample.y, dLogits, dFlat, dA, dZ, grads };
}
export function cnnPlan(model: CnnModel): CnnStep[] {
  const K = model.config.filters, filters = Array.from({ length: K }, (_, k) => k);
  return [
    { phase: 'input', stage: 'input' },
    ...filters.map(k => ({ phase: 'forward' as const, stage: 'conv' as const, k })),
    ...filters.map(k => ({ phase: 'forward' as const, stage: 'pool' as const, k })),
    { phase: 'forward', stage: 'dense' },
    { phase: 'loss', stage: 'loss' },
    { phase: 'backward', stage: 'dense' },
    ...filters.map(k => ({ phase: 'backward' as const, stage: 'pool' as const, k })),
    ...filters.map(k => ({ phase: 'backward' as const, stage: 'conv' as const, k })),
    ...Array.from({ length: CLASSES }, (_, c) => ({ phase: 'update' as const, stage: 'dense' as const, c, update: [`o${c}`] })),
    ...filters.map(k => ({ phase: 'update' as const, stage: 'conv' as const, k, update: [`f${k}`] })),
  ];
}
export const cnnLab = createLab<CnnModel, CnnImage, CnnTrace, CnnStep>({
  trace: cnnTrace,
  plan: cnnPlan,
  score(model, sample) {
    const { probs } = cnnForward(model, sample.pixels);
    return { loss: -Math.log(Math.max(probs[sample.y], 1e-300)), correct: Number(argmax(probs) === sample.y), total: 1 };
  },
});
export type CnnFrame = ReturnType<typeof cnnLab.createFrame>;
