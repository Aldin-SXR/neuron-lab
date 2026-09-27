import { format } from './engine';
import { UPDATE_GROUPS, type SeqFrame } from './sequence';
import type { SeqSelection } from './SeqView';
import { useI18n } from './i18n';
import { Formula, InspectorShell, Tip, inkColor, valueColor } from './ui';

const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
function ValueTable({ head, rows }: { head: string[]; rows: number[][] }) {
  return <div className="table-scroll"><table className="value-table"><thead><tr>{head.map(h => <th key={h}>{h}</th>)}</tr></thead><tbody>{rows.map((row, j) => <tr key={j}><th>{j + 1}</th>{row.map((v, k) => <td key={k}>{v.toFixed(2)}</td>)}</tr>)}</tbody></table></div>;
}
function Heatmap({ values, cols }: { values: number[]; cols: number }) {
  const scale = Math.max(1e-9, ...values.map(Math.abs));
  const numbers = cols <= 4;
  return <div className={`mini-grid ${numbers ? '' : 'compact'}`} style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>{values.map((v, i) => <span key={i} style={{ background: valueColor(v, scale), color: inkColor(v, scale) }} title={format(v, 5)}>{numbers ? v.toFixed(2) : ''}</span>)}</div>;
}
export function SeqInspector({ frame, selection, picked, symbols, rate }: { frame: SeqFrame; selection: SeqSelection | null; picked: boolean; symbols: string[]; rate: number }) {
  const { t } = useI18n();
  const T = t.seq.inspector;
  const step = frame.plan[frame.cursor], phase = step.phase;
  const { steps, back } = frame.trace;
  const lstm = frame.model.kind === 'lstm';
  const H = frame.model.hidden, V = frame.model.inputs;
  if (phase === 'update' && !picked) {
    const group = UPDATE_GROUPS[frame.model.kind].find(g => g.id === step.group)!;
    let largest = { name: '', from: 0, to: 0, d: -1 };
    for (const key of group.keys) frame.source.params[key].forEach((v, i) => {
      const d = Math.abs(frame.model.params[key][i] - v);
      if (d > largest.d) {
        const cols = key.startsWith('W') ? frame.source.params[key].length / (key === 'Wy' ? V : H) : 1;
        largest = { name: cols > 1 ? `${key}[${Math.floor(i / cols) + 1},${i % cols + 1}]` : `${key}[${i + 1}]`, from: v, to: frame.model.params[key][i], d };
      }
    });
    return <InspectorShell>
      <span className="eyebrow">{t.phases.update.toUpperCase()}</span><h3>{T.updateTitle}</h3><p>{T.updateText}</p>
      <Formula caption={t.common.sgd} math={<>W<sub>new</sub> = W<sub>old</sub> − η · Σₜ ∂L/∂Wₜ</>}>η = {rate}</Formula>
      {group.keys.map(key => <Formula neutral key={key} caption={T.matrix(key)}><Heatmap values={frame.trace.grads[key]} cols={key.startsWith('b') ? 1 : key === 'Wy' ? H : frame.trace.grads[key].length / H}/></Formula>)}
      <Tip>{largest.d > 0 ? T.largest(largest.name, format(largest.from, 4), format(largest.to, 4)) : t.common.unchanged} {t.common.gradientsFrozen}</Tip>
    </InspectorShell>;
  }
  if (!selection) return <InspectorShell><p>{T.selectHint}</p></InspectorShell>;
  const n = selection.t, s = steps[n], b = back[n];
  const kicker = <div className="inspector-kicker"><span className="eyebrow">{T.stepEyebrow(n + 1)}</span><span className="small-chip">x = “{symbols[s.x]}”</span></div>;
  const prediction = s.target === null ? <p className="fine-print">{t.seq.noTarget}</p> : <Formula neutral caption={T.predictionCaption} math="p = softmax(Wy · h + by)" result={<>−log p(“{symbols[s.target]}”) = {format(-Math.log(s.probs[s.target]), 4)}</>}>{symbols.map((sym, o) => <span key={o} className={o === s.target ? 'target-symbol' : ''}>{sym}: {format(s.probs[o], 3)}{o === s.target ? ` ← ${T.target}` : ''}<br/></span>)}</Formula>;
  if (phase === 'loss') {
    const scored = steps.map((v, k) => ({ v, k })).filter(({ v }) => v.target !== null);
    return <InspectorShell>{kicker}<h3>{T.lossTitle}</h3><p>{T.lossText}</p>
      <Formula caption={T.lossCaption} math="L = (1/N) Σₜ −log pₜ(yₜ)" result={<>L = {format(frame.trace.loss, 6)}</>}>{scored.map(({ v, k }) => <span key={k}>t{k + 1}: −log({format(v.probs[v.target!], 4)}) = {format(-Math.log(v.probs[v.target!]), 4)}<br/></span>)}</Formula></InspectorShell>;
  }
  if (phase === 'backward') {
    return <InspectorShell>{kicker}<h3>{T.backwardTitle}</h3><p>{T.backwardText}</p>
      <Formula caption={T.gradientCaption} math={lstm ? '∂L/∂c = ∂L/∂h · o · (1 − tanh²c) + ∂L/∂c₊₁ · f₊₁' : '∂L/∂z = ∂L/∂h · (1 − h²)'} result={<>{T.norm} = {format(norm(b.dh), 5)}</>}>
        <ValueTable head={[T.unit, '∂L/∂h', lstm ? '∂L/∂c' : '∂L/∂z']} rows={b.dh.map((d, j) => [d, lstm ? b.dc![j] : b.dz[j]])}/>
      </Formula>
      <Tip>{T.vanishing}</Tip></InspectorShell>;
  }
  if (lstm) {
    const g = s.gates!;
    return <InspectorShell>{kicker}<h3>{T.lstmTitle}</h3><p>{T.lstmText}</p>
      <Formula caption={T.formulas} math={<span className="formula-lines">f = σ(Wf·[x, h₋₁] + bf)<br/>i = σ(Wi·[x, h₋₁] + bi)<br/>g = tanh(Wg·[x, h₋₁] + bg)<br/>o = σ(Wo·[x, h₋₁] + bo)<br/>c = f ⊙ c₋₁ + i ⊙ g<br/>h = o ⊙ tanh(c)</span>}/>
      <Formula neutral caption={T.values}><ValueTable head={[T.unit, 'f', 'i', 'g', 'o', T.prevCell, T.newCell, 'h']} rows={s.h.map((h, j) => [g.f[j], g.i[j], g.g[j], g.o[j], s.cPrev[j], s.c![j], h])}/></Formula>
      {prediction}
    </InspectorShell>;
  }
  const p = frame.source.params;
  return <InspectorShell>{kicker}<h3>{T.forwardTitle}</h3><p>{T.forwardText}</p>
    <Formula caption={T.formulas} math="h = tanh(Wx·x + Wh·h₋₁ + b)"/>
    <Formula neutral caption={T.values}><ValueTable head={[T.unit, T.fromInput, T.fromMemory, 'b', T.sum, T.output]} rows={s.h.map((h, j) => [p.Wx[j * V + s.x], s.hPrev.reduce((sum, v, k) => sum + p.Wh[j * H + k] * v, 0), p.bh[j], s.z[j], h])}/></Formula>
    {prediction}
  </InspectorShell>;
}
