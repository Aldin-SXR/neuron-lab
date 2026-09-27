// Shared training machinery for the CNN, RNN, and LSTM engines. Each engine
// computes one sample's exact gradients up front; a lesson then walks through
// a plan of forward, loss, backward, and grouped parameter-update steps.
export type Phase = 'input' | 'forward' | 'loss' | 'backward' | 'update';
export interface Metric { epoch: number; loss: number; accuracy: number }
export interface LabFrame { plan: { phase: Phase }[]; cursor: number; sampleIndex: number; samplesSeen: number; history: Metric[]; metric: Metric; learningRate?: number }
export type Params = Record<string, number[]>;
export interface ParamModel { params: Params }
export interface Gradients { grads: Params; loss: number }
export interface PlanStep { phase: Phase; update?: string[] }
export interface Engine<M extends ParamModel, X, T extends Gradients, S extends PlanStep> {
  trace(model: M, sample: X): T;
  plan(model: M, sample: X): S[];
  score(model: M, sample: X): { loss: number; correct: number; total: number };
}
export interface ParamFrame<M, T, S extends PlanStep> extends LabFrame { model: M; source: M; trace: T; plan: S[] }

/** Thrown when parameters become non-finite or huge; the interface explains it in the active language. */
export class DivergedError extends Error { constructor() { super('diverged'); } }

export function softmax(z: number[]): number[] {
  const max = Math.max(...z);
  const exps = z.map(v => Math.exp(v - max));
  const sum = exps.reduce((s, v) => s + v, 0);
  return exps.map(v => v / sum);
}
export const sigmoid = (v: number) => v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v));
export const argmax = (values: number[]) => values.reduce((best, v, i) => v > values[best] ? i : best, 0);

export function createLab<M extends ParamModel, X, T extends Gradients, S extends PlanStep>(engine: Engine<M, X, T, S>) {
  type Frame = ParamFrame<M, T, S>;
  function metrics(model: M, data: X[], samplesSeen = 0): Metric {
    let loss = 0, correct = 0, total = 0;
    for (const sample of data) { const s = engine.score(model, sample); loss += s.loss; correct += s.correct; total += s.total; }
    return { epoch: samplesSeen / data.length, loss: loss / data.length, accuracy: correct / Math.max(total, 1) };
  }
  function createFrame(model: M, data: X[], sampleIndex = 0, samplesSeen = 0, history?: Metric[]): Frame {
    const metric = metrics(model, data, samplesSeen);
    const sample = data[sampleIndex];
    return { model, source: model, trace: engine.trace(model, sample), plan: engine.plan(model, sample), cursor: 0, sampleIndex, samplesSeen, metric, history: history ?? [metric] };
  }
  function applyUpdate(model: M, grads: Params, keys: string[], rate: number): M {
    const params = { ...model.params };
    for (const key of keys) {
      params[key] = model.params[key].map((v, i) => v - rate * grads[key][i]);
      if (params[key].some(v => !Number.isFinite(v) || Math.abs(v) > 1e6)) throw new DivergedError();
    }
    return { ...model, params };
  }
  function step(frame: Frame, data: X[], rate: number): Frame {
    const last = frame.plan.length - 1;
    if (frame.cursor === last) return createFrame(frame.model, data, (frame.sampleIndex + 1) % data.length, frame.samplesSeen, frame.history);
    rate = frame.learningRate ?? rate;
    const cursor = frame.cursor + 1;
    const event = frame.plan[cursor];
    const model = event.update ? applyUpdate(frame.model, frame.trace.grads, event.update, rate) : frame.model;
    const samplesSeen = frame.samplesSeen + Number(cursor === last);
    const metric = event.update ? metrics(model, data, samplesSeen) : frame.metric;
    const history = cursor === last ? [...frame.history, metric].slice(-400) : frame.history;
    return { ...frame, cursor, model, samplesSeen, metric, history, learningRate: rate };
  }
  function trainEpoch(frame: Frame, data: X[], rate: number): Frame {
    let current = frame;
    // Preserve partially completed lessons: finish their pending parameter updates.
    if (current.cursor > 0 && current.cursor < current.plan.length - 1) {
      while (current.cursor < current.plan.length - 1) current = step(current, data, rate);
    }
    const offset = current.cursor > 0 ? 1 : 0;
    let model = current.model;
    for (let n = 0; n < data.length; n++) {
      const t = engine.trace(model, data[(current.sampleIndex + offset + n) % data.length]);
      model = applyUpdate(model, t.grads, Object.keys(t.grads), rate);
    }
    const next = createFrame(model, data, (current.sampleIndex + offset) % data.length, current.samplesSeen + data.length);
    next.history = [...current.history, next.metric].slice(-400);
    return next;
  }
  function parameterCount(model: M) { return Object.values(model.params).reduce((s, p) => s + p.length, 0); }
  return { metrics, createFrame, step, trainEpoch, parameterCount };
}
