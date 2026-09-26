import { ArrowDown, ArrowRight, SlidersHorizontal, Sparkles } from 'lucide-react';
import { format, type Activation, type Frame } from './engine';
import type { Selection } from './NetworkView';

const activationFormula: Record<Activation, string> = { tanh: 'a = tanh(z)', sigmoid: 'a = 1 / (1 + e⁻ᶻ)', relu: 'a = max(0, z)', linear: 'a = z', softmax: 'aⱼ = eᶻʲ / Σₖ eᶻᵏ' };
const derivativeFormula: Record<Activation, string> = { tanh: 'f′(z) = 1 − a²', sigmoid: 'f′(z) = a(1 − a)', relu: 'f′(z) = 1 if z > 0, else 0', linear: 'f′(z) = 1', softmax: 'δⱼ = aⱼ(gⱼ − Σₖ gₖaₖ)' };
export function Inspector({ frame, selection, rate, onEdit }: { frame: Frame; selection: Selection | null; rate: number; onEdit: (s: Selection) => void }) {
  const event = frame.plan[frame.cursor];
  const selected = selection ?? { layer: event.layer ?? 0, neuron: event.neuron ?? 0, input: event.input, bias: event.bias };
  const { layer: l, neuron: j } = selected;
  if (l < 0) return <aside className="inspector card"><div className="panel-heading"><Sparkles size={16}/><h2>Inside the calculation</h2></div><div className="inspector-content"><span className="eyebrow">INPUT FEATURE</span><h3>Meet x{j + 1}</h3><div className="formula-box"><span className="math">x{j + 1} = {format(frame.trace.a[0][j])}</span></div><p>Inputs are the information we give the network. They have no trainable weights of their own. Each outgoing connection multiplies this value by a weight.</p><div className="tip"><Sparkles size={16}/><p>Select a neuron in the next layer to see where this input goes.</p></div></div></aside>;
  const layer = frame.trace.source.layers[l];
  const t = frame.trace;
  const isOutput = l === frame.network.layers.length - 1;
  const i = selected.input ?? 0;
  const bias = selected.bias ?? false;
  const old = bias ? layer.biases[j] : layer.weights[j][i];
  const gradient = bias ? t.db[l][j] : t.dw[l][j][i];
  const actual = bias ? frame.network.layers[l].biases[j] : frame.network.layers[l].weights[j][i];
  const phase = event.phase;
  const upstream = isOutput ? t.a[l + 1][j] - t.target[j] : frame.trace.source.layers[l + 1].weights.reduce((s, row, k) => s + row[j] * t.delta[l + 1][k], 0);
  const upstreamVector = t.a[l + 1].map((a, n) => isOutput ? a - t.target[n] : frame.trace.source.layers[l + 1].weights.reduce((s, row, k) => s + row[n] * t.delta[l + 1][k], 0));
  const derivative = layer.activation === 'tanh' ? 1 - t.a[l + 1][j] ** 2 : layer.activation === 'sigmoid' ? t.a[l + 1][j] * (1 - t.a[l + 1][j]) : layer.activation === 'relu' ? Number(t.z[l][j] > 0) : 1;
  return <aside className="inspector card">
    <div className="panel-heading"><Sparkles size={16}/><h2>Inside the calculation</h2></div>
    <div className="inspector-content">
      <div className="inspector-kicker"><span className="eyebrow">{isOutput ? 'OUTPUT LAYER' : `HIDDEN LAYER ${l + 1}`}</span><span className="small-chip">Neuron {j + 1}</span></div>
      <h3>{phase === 'backward' ? 'Trace the responsibility.' : phase === 'update' ? 'A small step toward better.' : phase === 'loss' ? 'How far off are we?' : 'A little math. A new signal.'}</h3>
      {phase === 'loss' ? <>
        <p>The loss measures the distance between our prediction and the target. Smaller is better.</p>
        <div className="formula-box"><div className="formula-caption">HALF SQUARED ERROR</div><div className="math">L = ½ Σⱼ (ŷⱼ − yⱼ)²</div><div className="numeric-formula">½ × ({t.a.at(-1)!.map((v, k) => `(${format(v)} − ${t.target[k]})²`).join(' + ')})</div><div className="formula-result">= {format(t.loss, 6)}</div></div>
        <div className="tip"><Sparkles size={16}/><p>This loss is for one example. The chart below averages the loss over the entire dataset.</p></div>
      </> : phase === 'backward' ? <>
        <p>How much does this neuron's weighted sum affect the loss? The chain rule connects the two.</p>
        <div className="formula-box"><div className="formula-caption">1 · ERROR SIGNAL</div><div className="math">{layer.activation === 'softmax' ? derivativeFormula.softmax : isOutput ? 'δ = (a − y) · f′(z)' : 'δⱼ = (Σₖ wₖⱼ δₖ) · f′(zⱼ)'}</div>{layer.activation !== 'softmax' ? <div className="numeric-formula">{derivativeFormula[layer.activation]}<br/>δ = {format(upstream)} × {format(derivative)}</div> : <div className="numeric-formula">δ = {format(t.a[l + 1][j])} × ({format(upstream)} − {format(upstreamVector.reduce((s, g, k) => s + g * t.a[l + 1][k], 0))})<br/>The sum includes every neuron in this layer.</div>}<div className="formula-result">δ = {format(t.delta[l][j], 6)}</div></div>
        <ArrowDown className="flow-arrow" size={17}/>
        <div className="formula-box neutral"><div className="formula-caption">2 · WEIGHT GRADIENT · INPUT {i + 1}</div><div className="math">∂L/∂w = δ · aᵖʳᵉᵛ</div><div className="numeric-formula">{format(t.delta[l][j])} × {format(t.a[l][i])}</div><div className="formula-result">= {format(t.dw[l][j][i], 6)}</div><div className="numeric-formula">Bias gradient ∂L/∂b = δ</div></div>
      </> : phase === 'update' ? <>
        <p>Move {bias ? 'the bias' : 'this weight'} against its gradient to reduce the loss. The learning rate controls the step size.</p>
        <div className="formula-box"><div className="formula-caption">STOCHASTIC GRADIENT DESCENT</div><div className="math">{bias ? 'b' : 'w'}<sub>new</sub> = {bias ? 'b' : 'w'}<sub>old</sub> − η · ∂L/∂{bias ? 'b' : 'w'}</div><div className="numeric-formula">{format(old)} − {rate} × {format(gradient)}</div><div className="formula-result">= {format(old - rate * gradient, 6)}</div></div>
        <div className="value-change"><div><span>Before sample</span><strong>{format(old)}</strong></div><ArrowRight size={17}/><div><span>Current value</span><strong className={old !== actual ? 'changed' : ''}>{format(actual)}</strong></div></div>
        <div className="tip"><Sparkles size={16}/><p>{old === actual ? 'This parameter is unchanged so far (or has a zero gradient).' : `Changed by ${format(actual - old, 6)}.`} Gradients use the weights from the start of this sample.</p></div>
      </> : <>
        <p>Gather the incoming signals, add a bias, then apply an activation function.</p>
        <div className="formula-box"><div className="formula-caption">1 · WEIGHTED SUM</div><div className="math">z = Σᵢ wᵢxᵢ + b</div><div className="numeric-formula weighted-terms">{layer.weights[j].map((w, k) => <span key={k}>{k > 0 ? '+ ' : ''}<b>{format(w, 3)}</b> × {format(t.a[l][k], 3)} </span>)}<span>+ {format(layer.biases[j], 3)}</span></div><div className="formula-result">z = {format(t.z[l][j])}</div></div>
        <ArrowDown className="flow-arrow" size={17}/>
        <div className="formula-box neutral"><div className="formula-caption">2 · {layer.activation.toUpperCase()} ACTIVATION</div><div className="math">{activationFormula[layer.activation]}</div><div className="formula-result">a = {format(t.a[l + 1][j])}</div></div>
        <p className="fine-print">{phase === 'input' ? 'Preview of the next calculation. Step forward to send the first signal.' : 'Values use the weights at the start of this example.'}{layer.activation === 'softmax' ? ' Softmax uses all logits in this layer together.' : ''}</p>
      </>}
      <button className="text-button edit-parameters" onClick={() => onEdit(selected)}><SlidersHorizontal size={14}/> Edit this neuron's parameters</button>
    </div>
  </aside>;
}
