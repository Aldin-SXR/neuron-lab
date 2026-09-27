import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ACTIVATIONS, DEFAULT_SPECS, RECOMMENDED_RATE, createFrame, createNetwork, dataset, metrics, step, trace, trainEpoch } from './engine';

test('every activation has correct weight and bias gradients, including the softmax Jacobian', () => {
  for (const activation of ACTIVATIONS) {
    for (const output of ACTIVATIONS) {
      const network = createNetwork([{ size: 3, activation }, { size: output === 'softmax' ? 2 : 1, activation: output }], 17);
      const sample = { x: [0.3, -0.7], y: 1 };
      const analytical = trace(network, sample);
      const epsilon = 1e-5;
      network.layers.forEach((layer, l) => layer.weights.forEach((row, j) => {
        row.forEach((value, i) => {
          row[i] = value + epsilon;
          const plus = trace(network, sample).loss;
          row[i] = value - epsilon;
          const minus = trace(network, sample).loss;
          row[i] = value;
          assert.ok(Math.abs((plus - minus) / (2 * epsilon) - analytical.dw[l][j][i]) < 1e-7, `${activation}/${output} weight gradient ${l},${j},${i}`);
        });
        const bias = layer.biases[j];
        layer.biases[j] = bias + epsilon;
        const plus = trace(network, sample).loss;
        layer.biases[j] = bias - epsilon;
        const minus = trace(network, sample).loss;
        layer.biases[j] = bias;
        assert.ok(Math.abs((plus - minus) / (2 * epsilon) - analytical.db[l][j]) < 1e-7);
      }));
    }
  }
});
test('a lesson changes only its highlighted parameter and leaves earlier frames intact', () => {
  const data = dataset('xor');
  let frame = createFrame(createNetwork(DEFAULT_SPECS), data);
  const original = structuredClone(frame);
  while (frame.cursor < frame.plan.length - 1) {
    const previous = frame;
    frame = step(frame, data, 0.3);
    const event = frame.plan[frame.cursor];
    if (event.phase !== 'update') assert.deepEqual(frame.network, original.network);
    else {
      const expected = structuredClone(previous.network);
      if (event.bias) expected.layers[event.layer!].biases[event.neuron!] -= 0.3 * original.trace.db[event.layer!][event.neuron!];
      else expected.layers[event.layer!].weights[event.neuron!][event.input!] -= 0.3 * original.trace.dw[event.layer!][event.neuron!][event.input!];
      assert.deepEqual(frame.network, expected);
    }
  }
  assert.equal(frame.samplesSeen, 1);
  assert.deepEqual(original.network, createNetwork(DEFAULT_SPECS));
  assert.deepEqual(original.trace.source, original.network);
});
test('default ANN learns XOR and the nonlinear circle dataset with actual SGD', () => {
  for (const problem of ['xor', 'circle'] as const) {
    const data = dataset(problem);
    let frame = createFrame(createNetwork(DEFAULT_SPECS), data);
    const initialLoss = frame.metric.loss;
    for (let epoch = 0; epoch < (problem === 'xor' ? 1500 : 400); epoch++) frame = trainEpoch(frame, data, RECOMMENDED_RATE[problem]);
    assert.ok(frame.metric.loss < initialLoss / 3, `${problem} loss: ${frame.metric.loss}`);
    assert.ok(frame.metric.accuracy >= 0.95, `${problem} accuracy: ${frame.metric.accuracy}`);
    assert.equal(frame.metric.epoch, problem === 'xor' ? 1500 : 400);
  }
});
test('train mode finishes pending lesson updates before starting its epoch', () => {
  const data = dataset('xor');
  let partial = createFrame(createNetwork(DEFAULT_SPECS), data);
  while (partial.plan[partial.cursor].phase !== 'update') partial = step(partial, data, 0.3);
  let completed = partial;
  while (completed.cursor < completed.plan.length - 1) completed = step(completed, data, 0.3);
  assert.deepEqual(trainEpoch(partial, data, 0.3), trainEpoch(completed, data, 0.3));
  assert.equal(metrics(partial.network, data).loss, partial.metric.loss);
});
