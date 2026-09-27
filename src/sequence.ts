// Recurrent networks (a simple tanh RNN and an LSTM) with explicit
// backpropagation through time. Inputs are one-hot symbols; outputs use softmax
// with cross-entropy averaged over the time steps that have a target.
import { random } from './engine';
import { argmax, createLab, sigmoid, softmax, type ParamModel, type Phase } from './lab';

export type SeqKind = 'rnn' | 'lstm';
export type SeqProblem = 'word' | 'memory';
export interface SeqModel extends ParamModel { kind: SeqKind; hidden: number; inputs: number; outputs: number }
export interface Sequence { x: number[]; y: (number | null)[] }
export interface SeqTask { problem: SeqProblem; symbols: string[]; data: Sequence[]; word?: string; length?: number }
export const GATES = ['f', 'i', 'g', 'o'] as const;
export type Gate = typeof GATES[number];
export interface SeqStepValues {
  x: number; hPrev: number[]; cPrev: number[];
  z: number[]; h: number[];
  gates?: Record<Gate, number[]>; c?: number[]; tanhC?: number[];
  logits: number[]; probs: number[]; target: number | null; loss: number;
}
export interface SeqBackValues { dLogits: number[]; dh: number[]; dz: number[]; dc?: number[]; dGates?: Record<Gate, number[]> }
export interface SeqTrace { steps: SeqStepValues[]; back: SeqBackValues[]; loss: number; grads: Record<string, number[]> }
export interface SeqStep { phase: Phase; t?: number; group?: string; update?: string[] }
export const DEFAULT_WORD = 'hello';

export function sequenceTask(problem: SeqProblem, word = DEFAULT_WORD, length = 4): SeqTask {
  if (problem === 'word') {
    const symbols = [...new Set(word)];
    const ids = [...word].map(ch => symbols.indexOf(ch));
    return { problem, symbols, word, data: [{ x: ids.slice(0, -1), y: ids.slice(1) }] };
  }
  const rng = random(90 + length);
  const count = Math.min(2 ** length, 32);
  const data = Array.from({ length: count }, (_, n) => {
    // Enumerate every sequence while there are at most 32; otherwise sample reproducibly, balancing the first bit.
    const x = count === 2 ** length ? Array.from({ length }, (_, t) => (n >> (length - 1 - t)) & 1) : [n % 2, ...Array.from({ length: length - 1 }, () => Number(rng() < 0.5))];
    return { x, y: x.map((_, t) => t === length - 1 ? x[0] : null) };
  });
  return { problem, symbols: ['0', '1'], length, data };
}
/** Validates a word the learner typed: 3–12 letters with at most 8 distinct symbols. */
export function cleanWord(input: string) {
  const word = input.trim().toLowerCase();
  return /^[\p{L}]{3,12}$/u.test(word) && new Set(word).size <= 8 && new Set(word).size >= 2 ? word : null;
}

export function createSequenceModel(kind: SeqKind, hidden: number, symbols: number, seed = 3): SeqModel {
  const rng = random(seed);
  const matrix = (rows: number, cols: number) => { const s = Math.sqrt(6 / (rows + cols)); return Array.from({ length: rows * cols }, () => (rng() * 2 - 1) * s); };
  const params: Record<string, number[]> = {};
  if (kind === 'rnn') {
    params.Wx = matrix(hidden, symbols); params.Wh = matrix(hidden, hidden); params.bh = Array(hidden).fill(0);
  } else {
    // Each gate reads [x_t, h_{t-1}]. The forget bias starts at 1 so memory is kept by default.
    for (const gate of GATES) { params[`W${gate}`] = matrix(hidden, symbols + hidden); params[`b${gate}`] = Array(hidden).fill(gate === 'f' ? 1 : 0); }
  }
  params.Wy = matrix(symbols, hidden); params.by = Array(symbols).fill(0);
  return { kind, hidden, inputs: symbols, outputs: symbols, params };
}
/** Multiply a row-major matrix (rows × cols) by a vector. */
export function matVec(m: number[], rows: number, cols: number, v: number[]) {
  return Array.from({ length: rows }, (_, r) => v.reduce((s, x, c) => s + m[r * cols + c] * x, 0));
}
export function seqForward(model: SeqModel, sequence: Sequence): SeqStepValues[] {
  const { hidden: H, inputs: V, params: p } = model;
  let h = Array(H).fill(0) as number[], c = Array(H).fill(0) as number[];
  const targets = sequence.y.filter(y => y !== null).length;
  return sequence.x.map((x, t) => {
    const hPrev = h, cPrev = c;
    let z: number[], gates: Record<Gate, number[]> | undefined, tanhC: number[] | undefined;
    if (model.kind === 'rnn') {
      z = Array.from({ length: H }, (_, j) => p.Wx[j * V + x] + p.bh[j] + hPrev.reduce((s, v, k) => s + p.Wh[j * H + k] * v, 0));
      h = z.map(Math.tanh);
    } else {
      const input = [...Array.from({ length: V }, (_, k) => Number(k === x)), ...hPrev];
      const pre = Object.fromEntries(GATES.map(g => [g, matVec(p[`W${g}`], H, V + H, input).map((v, j) => v + p[`b${g}`][j])])) as Record<Gate, number[]>;
      gates = { f: pre.f.map(sigmoid), i: pre.i.map(sigmoid), g: pre.g.map(Math.tanh), o: pre.o.map(sigmoid) };
      c = cPrev.map((v, j) => gates!.f[j] * v + gates!.i[j] * gates!.g[j]);
      tanhC = c.map(Math.tanh);
      h = tanhC.map((v, j) => gates!.o[j] * v);
      z = pre.g;
    }
    const logits = Array.from({ length: model.outputs }, (_, o) => h.reduce((s, v, j) => s + p.Wy[o * H + j] * v, p.by[o]));
    const probs = softmax(logits);
    const target = sequence.y[t];
    const loss = target === null ? 0 : -Math.log(Math.max(probs[target], 1e-300)) / targets;
    return { x, hPrev, cPrev, z, h, gates, c: model.kind === 'lstm' ? c : undefined, tanhC, logits, probs, target, loss };
  });
}
export function seqTrace(model: SeqModel, sequence: Sequence): SeqTrace {
  const { hidden: H, inputs: V, outputs: O, params: p } = model;
  const steps = seqForward(model, sequence);
  const targets = sequence.y.filter(y => y !== null).length;
  const grads = Object.fromEntries(Object.entries(p).map(([k, v]) => [k, Array(v.length).fill(0) as number[]]));
  const back: SeqBackValues[] = [];
  let dhNext = Array(H).fill(0) as number[], dcNext = Array(H).fill(0) as number[];
  for (let t = steps.length - 1; t >= 0; t--) {
    const s = steps[t];
    const dLogits = s.target === null ? Array(O).fill(0) as number[] : s.probs.map((v, o) => (v - Number(o === s.target)) / targets);
    dLogits.forEach((d, o) => { s.h.forEach((v, j) => { grads.Wy[o * H + j] += d * v; }); grads.by[o] += d; });
    const dh = Array.from({ length: H }, (_, j) => dLogits.reduce((sum, d, o) => sum + d * p.Wy[o * H + j], dhNext[j]));
    if (model.kind === 'rnn') {
      const dz = dh.map((d, j) => d * (1 - s.h[j] ** 2));
      dz.forEach((d, j) => { grads.Wx[j * V + s.x] += d; grads.bh[j] += d; s.hPrev.forEach((v, k) => { grads.Wh[j * H + k] += d * v; }); });
      dhNext = Array.from({ length: H }, (_, k) => dz.reduce((sum, d, j) => sum + d * p.Wh[j * H + k], 0));
      back[t] = { dLogits, dh, dz };
    } else {
      const g = s.gates!;
      const dc = dh.map((d, j) => d * g.o[j] * (1 - s.tanhC![j] ** 2) + dcNext[j]);
      const dGates: Record<Gate, number[]> = {
        f: dc.map((d, j) => d * s.cPrev[j] * g.f[j] * (1 - g.f[j])),
        i: dc.map((d, j) => d * g.g[j] * g.i[j] * (1 - g.i[j])),
        g: dc.map((d, j) => d * g.i[j] * (1 - g.g[j] ** 2)),
        o: dh.map((d, j) => d * s.tanhC![j] * g.o[j] * (1 - g.o[j])),
      };
      const input = [...Array.from({ length: V }, (_, k) => Number(k === s.x)), ...s.hPrev];
      const dInput = Array(V + H).fill(0) as number[];
      for (const gate of GATES) dGates[gate].forEach((d, j) => {
        grads[`b${gate}`][j] += d;
        input.forEach((v, k) => { grads[`W${gate}`][j * (V + H) + k] += d * v; dInput[k] += d * p[`W${gate}`][j * (V + H) + k]; });
      });
      dhNext = dInput.slice(V);
      dcNext = dc.map((d, j) => d * g.f[j]);
      back[t] = { dLogits, dh, dz: dGates.g, dc, dGates };
    }
  }
  return { steps, back, loss: steps.reduce((s, v) => s + v.loss, 0), grads };
}
export const UPDATE_GROUPS: Record<SeqKind, { id: string; keys: string[] }[]> = {
  rnn: [{ id: 'input', keys: ['Wx', 'bh'] }, { id: 'recurrent', keys: ['Wh'] }, { id: 'readout', keys: ['Wy', 'by'] }],
  lstm: [...GATES.map(g => ({ id: g, keys: [`W${g}`, `b${g}`] })), { id: 'readout', keys: ['Wy', 'by'] }],
};
export function seqPlan(model: SeqModel, sequence: Sequence): SeqStep[] {
  const T = sequence.x.length;
  return [
    { phase: 'input' },
    ...Array.from({ length: T }, (_, t) => ({ phase: 'forward' as const, t })),
    { phase: 'loss' },
    ...Array.from({ length: T }, (_, n) => ({ phase: 'backward' as const, t: T - 1 - n })),
    ...UPDATE_GROUPS[model.kind].map(g => ({ phase: 'update' as const, group: g.id, update: g.keys })),
  ];
}
export const seqLab = createLab<SeqModel, Sequence, SeqTrace, SeqStep>({
  trace: seqTrace,
  plan: seqPlan,
  score(model, sequence) {
    const steps = seqForward(model, sequence);
    const scored = steps.filter(s => s.target !== null);
    return { loss: steps.reduce((s, v) => s + v.loss, 0), correct: scored.filter(s => argmax(s.probs) === s.target).length, total: scored.length };
  },
});
export type SeqFrame = ReturnType<typeof seqLab.createFrame>;
/** Greedy continuation: feed a start symbol, then keep feeding back the network's most likely prediction. */
export function generate(model: SeqModel, start: number, length: number) {
  const out = [start];
  for (let n = 1; n < length; n++) {
    const steps = seqForward(model, { x: out, y: out.map(() => null) });
    out.push(argmax(steps.at(-1)!.probs));
  }
  return out;
}
