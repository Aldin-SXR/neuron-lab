import { format } from './engine';
import { KERNEL, MAP, POOLED, poolWindow, receptiveField, type CnnFrame } from './cnn';
import { cnnProgress, type CnnSelection } from './CnnView';
import { useI18n } from './i18n';
import { FlowArrow, Formula, InspectorShell, Tip, inkColor, valueColor } from './ui';

function MiniGrid({ values, cols, scale }: { values: number[]; cols: number; scale?: number }) {
  const s = scale ?? Math.max(1e-6, ...values.map(Math.abs));
  return <div className="mini-grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>{values.map((v, i) => <span key={i} style={{ background: valueColor(v, s), color: inkColor(v, s) }}>{format(v, 2)}</span>)}</div>;
}
export function CnnInspector({ frame, selection, pixels, rate, classes }: { frame: CnnFrame; selection: CnnSelection; pixels: number[]; rate: number; classes: string[] }) {
  const { t } = useI18n();
  const T = t.cnn.inspector;
  const tr = frame.trace, step = frame.plan[frame.cursor], phase = step.phase;
  const source = frame.source.params, current = frame.model.params;
  const { activation, pooling } = frame.model.config;
  // Gradients appear only once the lesson has reached the step that computes them.
  const reached = cnnProgress(frame, false);
  const changeNote = (after: number[], before: number[]) => { const d = Math.max(...after.map((v, i) => Math.abs(v - before[i]))); return d === 0 ? t.common.unchanged : t.common.largestChange(format(d, 5)); };
  if (selection.kind === 'image') return <InspectorShell><span className="eyebrow">{T.imageEyebrow}</span><h3>{T.imageTitle}</h3><p>{T.imageText}</p><MiniGrid values={pixels} cols={6} scale={1}/><Tip>{T.imageTip}</Tip></InspectorShell>;
  if (selection.kind === 'loss') return <InspectorShell><span className="eyebrow">{T.lossCaption}</span><h3>{T.lossTitle}</h3><p>{T.lossText}</p>
    <Formula caption={T.lossCaption} math="L = −log(p_target)" result={<>= {format(tr.loss, 6)}</>}>−log({format(tr.probs[tr.target])}) · {classes[tr.target]}</Formula></InspectorShell>;
  if (selection.kind === 'filter') {
    const k = selection.k, key = `f${k}`;
    const showGradient = reached.backConv(k);
    return <InspectorShell>
      <div className="inspector-kicker"><span className="eyebrow">{T.filterEyebrow(k + 1)}</span><span className="small-chip">3×3</span></div>
      <h3>{T.filterTitle}</h3><p>{showGradient ? T.filterGradientText : T.filterText}</p>
      <Formula caption={T.weights}><MiniGrid values={source[key].slice(0, 9)} cols={KERNEL}/><span className="grid-note">{T.bias} = {format(source[key][9], 3)}</span></Formula>
      {showGradient && <><FlowArrow/><Formula neutral caption={T.gradients} math="∂L/∂w = Σ δ · x"><MiniGrid values={tr.grads[key].slice(0, 9)} cols={KERNEL}/><span className="grid-note">∂L/∂b = Σ δ = {format(tr.grads[key][9], 4)}</span></Formula></>}
      {phase === 'update' && <><FlowArrow/><Formula caption={T.newWeights} math={<>w<sub>new</sub> = w<sub>old</sub> − η · ∂L/∂w</>}><MiniGrid values={current[key].slice(0, 9)} cols={KERNEL}/><span className="grid-note">η = {rate} · {T.bias} = {format(current[key][9], 3)}</span></Formula><Tip>{changeNote(current[key], source[key])} {t.common.gradientsFrozen}</Tip></>}
    </InspectorShell>;
  }
  if (selection.kind === 'map') {
    const { k, p } = selection, f = source[`f${k}`], r = Math.floor(p / MAP), c = p % MAP;
    const field = receptiveField(r, c);
    return <InspectorShell>
      <div className="inspector-kicker"><span className="eyebrow">{T.mapEyebrow(k + 1)}</span><span className="small-chip">{T.cell(r + 1, c + 1)}</span></div>
      <h3>{T.mapTitle}</h3><p>{T.mapText}</p>
      <Formula caption={T.convCaption} math="z = Σ w · x + b" result={<>z = {format(tr.z[k][p])}</>}><span className="weighted-terms">{field.map((idx, q) => <span key={q}>{q > 0 ? '+ ' : ''}<b>{format(f[q], 2)}</b> × {format(pixels[idx], 2)} </span>)}<span>+ {format(f[KERNEL * KERNEL], 2)}</span></span></Formula>
      <FlowArrow/>
      <Formula neutral caption={T.activationCaption(activation.toUpperCase())} math={activation === 'relu' ? 'a = max(0, z)' : 'a = tanh(z)'} result={<>a = {format(tr.a[k][p])}</>}/>
      {reached.backPool(k) && <Formula caption={T.mapGradient} math={activation === 'relu' ? '∂L/∂z = ∂L/∂a · [z > 0]' : '∂L/∂z = ∂L/∂a · (1 − a²)'} result={<>∂L/∂z = {format(tr.dZ[k][p], 5)}</>}>∂L/∂a = {format(tr.dA[k][p], 5)}<br/>{T.mapGradientText}</Formula>}
    </InspectorShell>;
  }
  if (selection.kind === 'pool') {
    const { k, p } = selection, window = poolWindow(Math.floor(p / POOLED), p % POOLED), max = pooling === 'max';
    const g = tr.dFlat[k * POOLED * POOLED + p];
    return <InspectorShell>
      <div className="inspector-kicker"><span className="eyebrow">{T.poolEyebrow(k + 1)}</span><span className="small-chip">{T.cell(Math.floor(p / POOLED) + 1, p % POOLED + 1)}</span></div>
      <h3>{max ? T.poolTitle : T.poolTitleAverage}</h3><p>{T.poolText(max)}</p>
      <Formula caption={pooling.toUpperCase()} math={max ? 'p = max(a₁, a₂, a₃, a₄)' : 'p = (a₁ + a₂ + a₃ + a₄) / 4'} result={<>p = {format(tr.pooled[k][p])}</>}>{max ? 'max' : 'avg'}({window.map(i => format(tr.a[k][i], 3)).join(', ')})</Formula>
      {reached.backDense && <Formula neutral caption={t.cnn.step.backPoolTitle(k + 1)} result={<>∂L/∂p = {format(g, 5)}</>}>{max ? t.cnn.step.backPoolMax : t.cnn.step.backPoolAverage}</Formula>}
    </InspectorShell>;
  }
  const c = selection.c, o = source[`o${c}`], n = tr.flat.length;
  const y = Number(c === tr.target);
  return <InspectorShell>
    <div className="inspector-kicker"><span className="eyebrow">{T.outputEyebrow}</span><span className="small-chip">{classes[c]}</span></div>
    <h3>{T.outputTitle(classes[c])}</h3><p>{T.outputText}</p>
    <Formula caption={T.scoreCaption} math="z = Σ v · p + b" result={<>z = {format(tr.logits[c])}</>}><span className="weighted-terms">{tr.flat.map((v, m) => <span key={m}>{m > 0 ? '+ ' : ''}<b>{format(o[m], 2)}</b> × {format(v, 2)} </span>)}<span>+ {format(o[n], 2)}</span></span></Formula>
    <FlowArrow/>
    <Formula neutral caption={T.softmaxCaption} math="pᶜ = eᶻᶜ / Σₖ eᶻᵏ" result={<>p = {format(tr.probs[c])}</>}>e^{format(tr.logits[c], 3)} / ({tr.logits.map(v => `e^${format(v, 2)}`).join(' + ')})</Formula>
    {reached.backDense && <Formula caption={T.gradientCaption} math="∂L/∂z = p − y" result={<>= {format(tr.dLogits[c], 5)}</>}>{format(tr.probs[c])} − {y}<br/>{T.gradientText}</Formula>}
    {phase === 'update' && <Tip>{T.updateText} {changeNote(current[`o${c}`], o)}</Tip>}
  </InspectorShell>;
}
