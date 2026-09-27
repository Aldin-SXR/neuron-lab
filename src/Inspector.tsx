import { ArrowRight, SlidersHorizontal } from 'lucide-react';
import { format, type Activation, type Frame } from './engine';
import { useI18n } from './i18n';
import type { Selection } from './NetworkView';
import { FlowArrow, Formula, InspectorShell, Tip } from './ui';

const activationFormula: Record<Activation, string> = { tanh: 'a = tanh(z)', sigmoid: 'a = 1 / (1 + e⁻ᶻ)', relu: 'a = max(0, z)', linear: 'a = z', softmax: 'aⱼ = eᶻʲ / Σₖ eᶻᵏ' };
const derivativeFormula: Record<Activation, string> = { tanh: 'f′(z) = 1 − a²', sigmoid: 'f′(z) = a(1 − a)', relu: 'f′(z) = 1 if z > 0, else 0', linear: 'f′(z) = 1', softmax: 'δⱼ = aⱼ(gⱼ − Σₖ gₖaₖ)' };
export function Inspector({ frame, selection, rate, onEdit }: { frame: Frame; selection: Selection | null; rate: number; onEdit: (s: Selection) => void }) {
  const { t } = useI18n();
  const T = t.ann.inspector;
  const event = frame.plan[frame.cursor];
  const selected = selection ?? { layer: event.layer ?? 0, neuron: event.neuron ?? 0, input: event.input, bias: event.bias };
  const { layer: l, neuron: j } = selected;
  if (l < 0) return <InspectorShell><span className="eyebrow">{T.inputEyebrow}</span><h3>{T.inputTitle(j + 1)}</h3><Formula caption="x" math={<>x{j + 1} = {format(frame.trace.a[0][j])}</>}/><p>{T.inputText}</p><Tip>{T.inputTip}</Tip></InspectorShell>;
  const layer = frame.trace.source.layers[l];
  const tr = frame.trace;
  const isOutput = l === frame.network.layers.length - 1;
  const i = selected.input ?? 0;
  const bias = selected.bias ?? false;
  const old = bias ? layer.biases[j] : layer.weights[j][i];
  const gradient = bias ? tr.db[l][j] : tr.dw[l][j][i];
  const actual = bias ? frame.network.layers[l].biases[j] : frame.network.layers[l].weights[j][i];
  const phase = event.phase;
  const upstreamVector = tr.a[l + 1].map((a, n) => isOutput ? a - tr.target[n] : tr.source.layers[l + 1].weights.reduce((s, row, k) => s + row[n] * tr.delta[l + 1][k], 0));
  const upstream = upstreamVector[j];
  const derivative = layer.activation === 'tanh' ? 1 - tr.a[l + 1][j] ** 2 : layer.activation === 'sigmoid' ? tr.a[l + 1][j] * (1 - tr.a[l + 1][j]) : layer.activation === 'relu' ? Number(tr.z[l][j] > 0) : 1;
  return <InspectorShell>
    <div className="inspector-kicker"><span className="eyebrow">{isOutput ? T.outputEyebrow : T.hiddenEyebrow(l + 1)}</span><span className="small-chip">{T.neuron(j + 1)}</span></div>
    <h3>{phase === 'backward' ? T.backwardTitle : phase === 'update' ? T.updateTitle : phase === 'loss' ? T.lossTitle : T.forwardTitle}</h3>
    {phase === 'loss' ? <>
      <p>{T.lossText}</p>
      <Formula caption={T.lossCaption} math="L = ½ Σⱼ (ŷⱼ − yⱼ)²" result={<>= {format(tr.loss, 6)}</>}>½ × ({tr.a.at(-1)!.map((v, k) => `(${format(v)} − ${tr.target[k]})²`).join(' + ')})</Formula>
      <Tip>{T.lossTip}</Tip>
    </> : phase === 'backward' ? <>
      <p>{T.backwardText}</p>
      <Formula caption={T.errorSignal} math={layer.activation === 'softmax' ? derivativeFormula.softmax : isOutput ? 'δ = (a − y) · f′(z)' : 'δⱼ = (Σₖ wₖⱼ δₖ) · f′(zⱼ)'} result={<>δ = {format(tr.delta[l][j], 6)}</>}>
        {layer.activation !== 'softmax' ? <>{layer.activation === 'relu' ? T.reluDerivative : derivativeFormula[layer.activation]}<br/>δ = {format(upstream)} × {format(derivative)}</> : <>δ = {format(tr.a[l + 1][j])} × ({format(upstream)} − {format(upstreamVector.reduce((s, g, k) => s + g * tr.a[l + 1][k], 0))})<br/>{T.softmaxSum}</>}
      </Formula>
      <FlowArrow/>
      <Formula neutral caption={T.weightGradient(i + 1)} math="∂L/∂w = δ · aᵖʳᵉᵛ" result={<>= {format(tr.dw[l][j][i], 6)}</>}>{format(tr.delta[l][j])} × {format(tr.a[l][i])}<br/>{T.biasGradient}</Formula>
    </> : phase === 'update' ? <>
      <p>{T.updateText(bias)}</p>
      <Formula caption={t.common.sgd} math={<>{bias ? 'b' : 'w'}<sub>new</sub> = {bias ? 'b' : 'w'}<sub>old</sub> − η · ∂L/∂{bias ? 'b' : 'w'}</>} result={<>= {format(old - rate * gradient, 6)}</>}>{format(old)} − {rate} × {format(gradient)}</Formula>
      <div className="value-change"><div><span>{t.common.before}</span><strong>{format(old)}</strong></div><ArrowRight size={18}/><div><span>{t.common.current}</span><strong className={old !== actual ? 'changed' : ''}>{format(actual)}</strong></div></div>
      <Tip>{old === actual ? t.common.unchanged : t.common.changedBy(format(actual - old, 6))} {t.common.gradientsFrozen}</Tip>
    </> : <>
      <p>{T.forwardText}</p>
      <Formula caption={T.weightedSum} math="z = Σᵢ wᵢxᵢ + b" result={<>z = {format(tr.z[l][j])}</>}><span className="weighted-terms">{layer.weights[j].map((w, k) => <span key={k}>{k > 0 ? '+ ' : ''}<b>{format(w, 3)}</b> × {format(tr.a[l][k], 3)} </span>)}<span>+ {format(layer.biases[j], 3)}</span></span></Formula>
      <FlowArrow/>
      <Formula neutral caption={T.activation(layer.activation.toUpperCase())} math={activationFormula[layer.activation]} result={<>a = {format(tr.a[l + 1][j])}</>}/>
      <p className="fine-print">{phase === 'input' ? T.preview : T.startValues}{layer.activation === 'softmax' ? T.softmaxNote : ''}</p>
    </>}
    <button className="text-button edit-parameters" onClick={() => onEdit(selected)}><SlidersHorizontal size={15}/> {T.editButton}</button>
  </InspectorShell>;
}
