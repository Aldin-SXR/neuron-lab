import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_CNN, cnnDataset, cnnLab, cnnTrace, createCnn, type CnnModel } from './cnn';
import { createSequenceModel, generate, seqLab, seqTrace, sequenceTask, cleanWord, type SeqModel } from './sequence';
import type { Params } from './lab';
import { format } from './engine';
import { en } from './locales/en';
import { bs } from './locales/bs';

/** Compare every analytical gradient with a central finite difference of the loss. */
function checkGradients(params: Params, analytical: Params, loss: () => number, label: string) {
  const epsilon = 1e-5;
  for (const [key, values] of Object.entries(params)) values.forEach((value, i) => {
    values[i] = value + epsilon; const plus = loss();
    values[i] = value - epsilon; const minus = loss();
    values[i] = value;
    assert.ok(Math.abs((plus - minus) / (2 * epsilon) - analytical[key][i]) < 1e-7, `${label} ${key}[${i}]`);
  });
}
test('CNN filter and dense gradients match finite differences for every activation and pooling', () => {
  for (const activation of ['relu', 'tanh'] as const) for (const pooling of ['max', 'average'] as const) for (const problem of ['lines', 'shapes'] as const) {
    const model = createCnn({ filters: 3, activation, pooling }, 11);
    const sample = cnnDataset(problem)[5];
    checkGradients(model.params, cnnTrace(model, sample).grads, () => cnnTrace(model, sample).loss, `${activation}/${pooling}/${problem}`);
  }
});
test('RNN and LSTM backpropagation through time matches finite differences', () => {
  for (const kind of ['rnn', 'lstm'] as const) for (const task of [sequenceTask('word', 'banana'), sequenceTask('memory', '', 5)]) {
    const model = createSequenceModel(kind, 3, task.symbols.length, 5);
    const sample = task.data[task.data.length - 1];
    checkGradients(model.params, seqTrace(model, sample).grads, () => seqTrace(model, sample).loss, `${kind}/${task.problem}`);
  }
});
test('a lesson updates exactly one parameter group per update step, using the gradients from the sample start', () => {
  const cases: [typeof cnnLab | typeof seqLab, CnnModel | SeqModel, unknown[]][] = [
    [cnnLab, createCnn(DEFAULT_CNN), cnnDataset('lines')],
    [seqLab, createSequenceModel('lstm', 3, 4), sequenceTask('word').data],
  ];
  for (const [lab, model, data] of cases) {
    const run = lab as unknown as typeof cnnLab;
    let frame = run.createFrame(model as CnnModel, data as never);
    const original = structuredClone(frame);
    while (frame.cursor < frame.plan.length - 1) {
      const previous = frame;
      frame = run.step(frame, data as never, 0.2);
      const event = frame.plan[frame.cursor];
      for (const key of Object.keys(model.params)) {
        const expected = event.update?.includes(key) ? previous.model.params[key].map((v, i) => v - 0.2 * original.trace.grads[key][i]) : previous.model.params[key];
        assert.deepEqual(frame.model.params[key], expected, `${key} at step ${frame.cursor}`);
      }
    }
    assert.equal(frame.samplesSeen, 1);
    assert.deepEqual(original.model, model);
    const next = run.step(frame, data as never, 0.2);
    assert.equal(next.cursor, 0);
    assert.equal(next.sampleIndex, 1 % data.length);
  }
});
test('the CNN learns both image problems with SGD', () => {
  for (const problem of ['lines', 'shapes'] as const) {
    const data = cnnDataset(problem);
    let frame = cnnLab.createFrame(createCnn(DEFAULT_CNN), data);
    const initial = frame.metric.loss;
    for (let epoch = 0; epoch < 60; epoch++) frame = cnnLab.trainEpoch(frame, data, 0.1);
    assert.ok(frame.metric.loss < initial / 4, `${problem} loss ${frame.metric.loss}`);
    assert.ok(frame.metric.accuracy >= 0.95, `${problem} accuracy ${frame.metric.accuracy}`);
  }
});
test('RNN and LSTM learn to spell a word and to remember the first bit', () => {
  for (const [kind, rate] of [['rnn', 0.3], ['lstm', 0.5]] as const) {
    const word = sequenceTask('word', 'hello');
    let frame = seqLab.createFrame(createSequenceModel(kind, 4, word.symbols.length), word.data);
    for (let epoch = 0; epoch < 300; epoch++) frame = seqLab.trainEpoch(frame, word.data, rate);
    assert.equal(frame.metric.accuracy, 1, `${kind} word accuracy`);
    assert.equal(generate(frame.model, 0, 5).map(i => word.symbols[i]).join(''), 'hello');
    const memory = sequenceTask('memory', '', 4);
    let recall = seqLab.createFrame(createSequenceModel(kind, 4, 2), memory.data);
    for (let epoch = 0; epoch < 200; epoch++) recall = seqLab.trainEpoch(recall, memory.data, rate);
    assert.equal(recall.metric.accuracy, 1, `${kind} memory accuracy`);
  }
});
test('train mode finishes pending lesson updates before starting its epoch', () => {
  const data = cnnDataset('shapes');
  let partial = cnnLab.createFrame(createCnn(DEFAULT_CNN), data);
  while (partial.plan[partial.cursor].phase !== 'update') partial = cnnLab.step(partial, data, 0.1);
  let completed = partial;
  while (completed.cursor < completed.plan.length - 1) completed = cnnLab.step(completed, data, 0.1);
  assert.deepEqual(cnnLab.trainEpoch(partial, data, 0.1), cnnLab.trainEpoch(completed, data, 0.1));
});
test('memory sequences are balanced and word input is validated', () => {
  for (const length of [2, 5, 8]) {
    const task = sequenceTask('memory', '', length);
    assert.equal(task.data.filter(s => s.x[0] === 1).length, task.data.length / 2);
    assert.ok(task.data.every(s => s.y.at(-1) === s.x[0] && s.y.slice(0, -1).every(y => y === null)));
    assert.equal(new Set(task.data.map(s => s.x.join(''))).size, task.data.length, `length ${length} has duplicates`);
  }
  assert.equal(cleanWord(' Neuron '), 'neuron');
  assert.equal(cleanWord('šećer'), 'šećer');
  assert.equal(cleanWord('aa'), null);
  assert.equal(cleanWord('aaaa'), null);
  assert.equal(cleanWord('abc1'), null);
  assert.equal(cleanWord('c\u030Cevapi'), 'čevapi', 'a letter typed with a separate accent mark');
  assert.equal(format(-0.004, 1), '0.0');
  assert.equal(format(-0.04, 1), '0.0');
  assert.equal(format(-0.5, 1), '-0.5');
});
test('the Bosnian translation has the same structure as English and no empty text', () => {
  function compare(a: unknown, b: unknown, path: string) {
    assert.equal(typeof a, typeof b, path);
    if (typeof a === 'string') { assert.ok((b as string).trim().length > 0, `${path} is empty`); return; }
    if (typeof a === 'function') { assert.equal(typeof (b as (...x: unknown[]) => unknown)(1, 2, 3, '4'), 'string', path); return; }
    if (Array.isArray(a)) { assert.equal(a.length, (b as unknown[]).length, path); a.forEach((v, i) => compare(v, (b as unknown[])[i], `${path}[${i}]`)); return; }
    const keys = Object.keys(a as object);
    assert.deepEqual(Object.keys(b as object).sort(), [...keys].sort(), path);
    for (const key of keys) compare((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key], `${path}.${key}`);
  }
  compare(en, bs, 'messages');
  assert.equal(bs.common.examples(1), '1 primjer');
  assert.equal(bs.common.layers(3), '3 sloja');
  assert.equal(bs.common.parameters(12), '12 parametara');
  assert.equal(bs.common.neurons(21), '21 neuron');
  assert.equal(bs.seq.hiddenCount(1), '1 skrivena jedinica');
  assert.equal(bs.seq.hiddenCount(3), '3 skrivene jedinice');
  assert.equal(en.seq.hiddenCount(1), '1 hidden unit');
  assert.match(bs.ann.step.trainText(4), /na svim primjerima \(4 primjera\)/);
  assert.match(bs.cnn.step.trainText(24), /\(24 slike\)/);
  assert.match(bs.cnn.step.trainText(48), /\(48 slika\)/);
  assert.match(bs.seq.step.trainText(32), /\(32 niza\)/);
  assert.match(bs.cnn.step.updateDenseText(4), /^Sve 4 težine/);
  assert.match(bs.cnn.step.updateDenseText(8), /^Svih 8 težina/);
});
