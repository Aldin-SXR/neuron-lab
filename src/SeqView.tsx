import type { CSSProperties } from 'react';
import { format } from './engine';
import type { SeqFrame } from './sequence';
import { useI18n } from './i18n';
import { ZoomControls, useZoom, valueColor } from './ui';

export interface SeqSelection { t: number }
const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

export function defaultSeqSelection(frame: SeqFrame): SeqSelection | null {
  const step = frame.plan[frame.cursor];
  if (step.t !== undefined) return { t: step.t };
  if (step.phase === 'input') return { t: 0 };
  if (step.phase === 'loss') { const steps = frame.trace.steps; return { t: steps.map(s => s.target !== null).lastIndexOf(true) }; }
  return null;
}
export function SeqView({ frame, symbols, selection, onSelect, training }: { frame: SeqFrame; symbols: string[]; selection: SeqSelection | null; onSelect: (s: SeqSelection) => void; training: boolean }) {
  const { t } = useI18n();
  const T = t.seq;
  const zoom = useZoom();
  const steps = frame.trace.steps, back = frame.trace.back;
  const lstm = frame.model.kind === 'lstm';
  const done = training ? frame.plan : frame.plan.slice(0, frame.cursor + 1);
  const forwardDone = (n: number) => done.some(s => s.phase === 'forward' && s.t === n);
  const backwardDone = (n: number) => !training && done.some(s => s.phase === 'backward' && s.t === n);
  const phase = training ? 'train' : frame.plan[frame.cursor].phase;
  const gradNorms = back.map(b => norm(b.dh));
  const gradMax = Math.max(1e-9, ...gradNorms);
  const cScale = lstm ? Math.max(1, ...steps.flatMap(s => s.c!.map(Math.abs))) : 1;
  const rows = ['output', ...(lstm ? ['gates', 'cell'] : []), 'hidden', 'input', 'gradient'];
  const style = { gridTemplateColumns: `minmax(6.5rem, auto) repeat(${steps.length}, minmax(${6.2 * zoom.zoom}rem, 1fr))`, gridTemplateRows: `repeat(${rows.length}, auto)`, '--square': `${1.15 * zoom.zoom}rem` } as CSSProperties;
  return <div className="network-stage" data-tour="network">
    <div className="network-scroll">
      <div className={`seq-grid phase-${phase}`} style={style}>
        {rows.map((row, r) => <div key={row} className={`row-label row-${row}`} style={{ gridRow: r + 1 }}>{T.rows[row as keyof typeof T.rows]}</div>)}
        {steps.map((s, n) => {
          const shown = forwardDone(n);
          const current = !training && frame.plan[frame.cursor].t === n;
          const selected = selection?.t === n;
          return <button key={n} className={`time-column ${selected ? 'selected' : ''} ${current ? 'current' : ''}`} style={{ gridColumn: n + 2, gridRow: `1 / span ${rows.length}` }} onClick={() => onSelect({ t: n })} aria-label={T.stepAria(n + 1, symbols[s.x])}>
            <div className="seq-cell row-output">{s.target === null ? <span className="no-target">{T.noTarget}</span> : symbols.map((sym, o) => <span key={o} className={`prob-row ${o === s.target ? 'target' : ''}`}><span className="prob-symbol">{sym}</span><span className="bar-track"><span className="bar-fill" style={{ width: shown ? `${s.probs[o] * 100}%` : 0 }}/></span><span className="prob-value">{shown ? Math.round(s.probs[o] * 100) : '·'}</span></span>)}</div>
            {lstm && <div className="seq-cell row-gates">{(['f', 'i', 'o'] as const).map(g => <span key={g} className="squares">{s.gates![g].map((v, j) => <span key={j} className="square" style={shown ? { background: `color-mix(in srgb, #3f9a7f ${Math.round(v * 100)}%, white)` } : undefined} title={shown ? `${g}${j + 1} = ${format(v, 3)}` : undefined}/>)}</span>)}</div>}
            {lstm && <div className="seq-cell row-cell recurrent"><span className="squares">{s.c!.map((v, j) => <span key={j} className="square" style={shown ? { background: valueColor(v, cScale) } : undefined} title={shown ? `c${j + 1} = ${format(v, 3)}` : undefined}/>)}</span></div>}
            <div className="seq-cell row-hidden recurrent"><span className="squares">{s.h.map((v, j) => <span key={j} className="square" style={shown ? { background: valueColor(v, 1) } : undefined} title={shown ? `h${j + 1} = ${format(v, 3)}` : undefined}/>)}</span></div>
            <div className="seq-cell row-input"><span className="token">{symbols[s.x]}</span><small>{T.stepLabel(n + 1)}</small></div>
            <div className="seq-cell row-gradient">{backwardDone(n) ? <><span className="bar-track"><span className="bar-fill gradient" style={{ width: `${gradNorms[n] / gradMax * 100}%` }}/></span><small>{format(gradNorms[n], 3)}</small></> : <small className="muted">—</small>}</div>
          </button>;
        })}
      </div>
    </div>
    <div className="canvas-footer"><span><span className="tiny-dot purple"/> {t.common.positive} <span className="tiny-dot sand"/> {t.common.negative}</span><span className="canvas-hint">{T.unrolled}</span><ZoomControls zoom={zoom}/></div>
  </div>;
}
