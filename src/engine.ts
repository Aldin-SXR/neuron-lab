// An intentionally small, framework-independent ANN engine. All gradients are
// explicit so the visualizer can explain exactly the calculation being used.
export type Activation = 'linear' | 'tanh' | 'sigmoid' | 'relu' | 'softmax';
export type Problem = 'xor' | 'circle' | 'diagonal';
export interface LayerSpec { size: number; activation: Activation }
export interface Layer extends LayerSpec { weights: number[][]; biases: number[] }
export interface Network { layers: Layer[] }
export interface Sample { x: number[]; y: number }
export interface Trace { source: Network; a: number[][]; z: number[][]; delta: number[][]; dw: number[][][]; db: number[][]; loss: number; target: number[] }
export interface Step { phase: 'input' | 'forward' | 'loss' | 'backward' | 'update'; layer?: number; neuron?: number; input?: number; bias?: boolean }
export interface Metric { epoch: number; loss: number; accuracy: number }
export interface Frame {
  network: Network; trace: Trace; plan: Step[]; cursor: number;
  sampleIndex: number; samplesSeen: number; history: Metric[]; metric: Metric;
  learningRate?: number;
}
export const DEFAULT_SPECS: LayerSpec[] = [{ size: 4, activation: 'tanh' }, { size: 3, activation: 'tanh' }, { size: 1, activation: 'sigmoid' }];
export const ACTIVATIONS: Activation[] = ['tanh', 'sigmoid', 'relu', 'linear', 'softmax'];

export function random(seed: number) {
  let value = seed >>> 0;
  return () => { value += 0x6D2B79F5; let t = value; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function dataset(problem: Problem): Sample[] {
  if (problem === 'xor') return [{ x: [0, 0], y: 0 }, { x: [0, 1], y: 1 }, { x: [1, 0], y: 1 }, { x: [1, 1], y: 0 }];
  const rng = random(28);
  return Array.from({ length: 80 }, () => {
    const x = [rng() * 2 - 1, rng() * 2 - 1];
    return { x, y: Number(problem === 'circle' ? x[0] ** 2 + x[1] ** 2 < 0.52 : x[0] + x[1] > 0) };
  });
}
export function createNetwork(specs: LayerSpec[], seed = 42): Network {
  const rng = random(seed);
  let inputs = 2;
  return { layers: specs.map(spec => {
    const scale = Math.sqrt(6 / (inputs + spec.size));
    const layer = { ...spec, weights: Array.from({ length: spec.size }, () => Array.from({ length: inputs }, () => (rng() * 2 - 1) * scale)), biases: Array(spec.size).fill(0) as number[] };
    inputs = spec.size;
    return layer;
  }) };
}
export function activate(z: number[], kind: Activation): number[] {
  if (kind === 'softmax') {
    const exps = z.map(v => Math.exp(v - Math.max(...z)));
    const sum = exps.reduce((s, v) => s + v, 0);
    return exps.map(v => v / sum);
  }
  return z.map(v => kind === 'tanh' ? Math.tanh(v) : kind === 'sigmoid' ? (v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v))) : kind === 'relu' ? Math.max(0, v) : v);
}
export function activationBackward(upstream: number[], a: number[], z: number[], kind: Activation): number[] {
  // Softmax has a full Jacobian: dL/dz_i = a_i(g_i - sum_j(g_j a_j)).
  if (kind === 'softmax') {
    const dot = upstream.reduce((s, g, i) => s + g * a[i], 0);
    return a.map((v, i) => v * (upstream[i] - dot));
  }
  return upstream.map((g, i) => g * (kind === 'tanh' ? 1 - a[i] ** 2 : kind === 'sigmoid' ? a[i] * (1 - a[i]) : kind === 'relu' ? Number(z[i] > 0) : 1));
}
export function forward(network: Network, x: number[]) {
  const a = [x.slice()];
  const z: number[][] = [];
  network.layers.forEach((layer, l) => {
    z.push(layer.weights.map((row, j) => row.reduce((s, w, i) => s + w * a[l][i], layer.biases[j])));
    a.push(activate(z[l], layer.activation));
  });
  return { a, z };
}
export function trace(network: Network, sample: Sample): Trace {
  const { a, z } = forward(network, sample.x);
  const output = a.at(-1)!;
  const target = output.length === 2 ? [1 - sample.y, sample.y] : [sample.y];
  // Half squared error is defined for every selectable output activation.
  const loss = output.reduce((s, v, i) => s + 0.5 * (v - target[i]) ** 2, 0);
  const delta: number[][] = [], dw: number[][][] = [], db: number[][] = [];
  for (let l = network.layers.length - 1; l >= 0; l--) {
    const layer = network.layers[l];
    const upstream = l === network.layers.length - 1 ? output.map((v, i) => v - target[i]) : a[l + 1].map((_, j) => network.layers[l + 1].weights.reduce((s, row, k) => s + row[j] * delta[l + 1][k], 0));
    delta[l] = activationBackward(upstream, a[l + 1], z[l], layer.activation);
    dw[l] = delta[l].map(d => a[l].map(v => d * v));
    db[l] = delta[l].slice();
  }
  return { source: network, a, z, delta, dw, db, loss, target };
}
export function probability(network: Network, x: number[]) {
  const output = forward(network, x).a.at(-1)!;
  return output.length === 2 ? output[1] : output[0];
}
export function metrics(network: Network, data: Sample[], samplesSeen = 0): Metric {
  let loss = 0, correct = 0;
  for (const sample of data) {
    const output = forward(network, sample.x).a.at(-1)!;
    const target = output.length === 2 ? [1 - sample.y, sample.y] : [sample.y];
    loss += output.reduce((s, v, i) => s + 0.5 * (v - target[i]) ** 2, 0);
    const prediction = output.length === 2 ? Number(output[1] > output[0]) : Number(output[0] >= 0.5);
    correct += Number(prediction === sample.y);
  }
  return { epoch: samplesSeen / data.length, loss: loss / data.length, accuracy: correct / data.length };
}
export function plan(network: Network): Step[] {
  const result: Step[] = [{ phase: 'input' }];
  network.layers.forEach((layer, l) => layer.weights.forEach((_, j) => result.push({ phase: 'forward', layer: l, neuron: j })));
  result.push({ phase: 'loss' });
  for (let l = network.layers.length - 1; l >= 0; l--) network.layers[l].weights.forEach((_, j) => result.push({ phase: 'backward', layer: l, neuron: j }));
  network.layers.forEach((layer, l) => layer.weights.forEach((row, j) => {
    row.forEach((_, i) => result.push({ phase: 'update', layer: l, neuron: j, input: i }));
    result.push({ phase: 'update', layer: l, neuron: j, bias: true });
  }));
  return result;
}
export function createFrame(network: Network, data: Sample[], sampleIndex = 1, samplesSeen = 0, history?: Metric[]): Frame {
  const metric = metrics(network, data, samplesSeen);
  return { network, trace: trace(network, data[sampleIndex]), plan: plan(network), cursor: 0, sampleIndex, samplesSeen, metric, history: history ?? [metric] };
}
function checked(network: Network) {
  if (network.layers.some(l => [...l.biases, ...l.weights.flat()].some(v => !Number.isFinite(v) || Math.abs(v) > 1e8))) throw new Error('The weights grew too large. Reduce the learning rate or reset the network.');
  return network;
}
export function step(frame: Frame, data: Sample[], learningRate: number): Frame {
  if (frame.cursor === frame.plan.length - 1) return createFrame(frame.network, data, (frame.sampleIndex + 1) % data.length, frame.samplesSeen, frame.history);
  learningRate = frame.learningRate ?? learningRate;
  const cursor = frame.cursor + 1;
  const event = frame.plan[cursor];
  let network = frame.network;
  if (event.phase === 'update') {
    network = structuredClone(network);
    const l = event.layer!, j = event.neuron!;
    if (event.bias) network.layers[l].biases[j] -= learningRate * frame.trace.db[l][j];
    else network.layers[l].weights[j][event.input!] -= learningRate * frame.trace.dw[l][j][event.input!];
    checked(network);
  }
  const samplesSeen = frame.samplesSeen + Number(cursor === frame.plan.length - 1);
  const metric = event.phase === 'update' ? metrics(network, data, samplesSeen) : frame.metric;
  const history = cursor === frame.plan.length - 1 ? [...frame.history, metric].slice(-400) : frame.history;
  return { ...frame, cursor, network, samplesSeen, metric, history, learningRate };
}
export function trainEpoch(frame: Frame, data: Sample[], learningRate: number): Frame {
  let current = frame;
  // Preserve partially completed lessons: finish their pending parameter updates.
  if (current.cursor > 0 && current.cursor < current.plan.length - 1) {
    while (current.cursor < current.plan.length - 1) current = step(current, data, learningRate);
  }
  let network = structuredClone(current.network);
  for (let n = 0; n < data.length; n++) {
    const sample = data[(current.sampleIndex + (current.cursor > 0 ? 1 : 0) + n) % data.length];
    const t = trace(network, sample);
    network.layers.forEach((layer, l) => layer.weights.forEach((row, j) => {
      row.forEach((_, i) => { row[i] -= learningRate * t.dw[l][j][i]; });
      layer.biases[j] -= learningRate * t.db[l][j];
    }));
    network = checked(network);
  }
  const next = createFrame(network, data, (current.sampleIndex + Number(current.cursor > 0)) % data.length, current.samplesSeen + data.length);
  next.history = [...current.history, next.metric].slice(-400);
  return next;
}
export function parameterCount(network: Network) { return network.layers.reduce((s, l) => s + l.weights.flat().length + l.biases.length, 0); }
export function format(value: number, digits = 4): string { return Math.abs(value) > 1e5 || (Math.abs(value) < 0.0001 && value !== 0) ? value.toExponential(2) : value.toFixed(digits); }
