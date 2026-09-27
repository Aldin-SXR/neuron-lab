import type { CSSProperties } from 'react';
import { format } from './engine';
import { CLASSES, IMAGE, KERNEL, MAP, POOLED, poolWindow, receptiveField, type CnnFrame } from './cnn';
import { useI18n } from './i18n';
import { ZoomControls, inkColor, useZoom, valueColor } from './ui';

export type CnnSelection = { kind: 'image' } | { kind: 'loss' } | { kind: 'filter'; k: number } | { kind: 'map'; k: number; p: number } | { kind: 'pool'; k: number; p: number } | { kind: 'output'; c: number };

/** Which parts of the network the lesson has reached so far (everything, while training). */
export function cnnProgress(frame: CnnFrame, training: boolean) {
  const done = training ? frame.plan : frame.plan.slice(0, frame.cursor + 1);
  const has = (phase: string, stage: string, k?: number) => done.some(s => s.phase === phase && s.stage === stage && (k === undefined || s.k === k));
  return {
    conv: (k: number) => has('forward', 'conv', k), pool: (k: number) => has('forward', 'pool', k), dense: has('forward', 'dense'),
    backPool: (k: number) => !training && has('backward', 'pool', k), backConv: (k: number) => !training && has('backward', 'conv', k), backDense: !training && has('backward', 'dense'),
  };
}
export function defaultCnnSelection(frame: CnnFrame): CnnSelection {
  const s = frame.plan[frame.cursor], tr = frame.trace;
  const strongest = (values: number[]) => values.reduce((b, v, i) => Math.abs(v) > Math.abs(values[b]) ? i : b, 0);
  if (s.stage === 'input') return { kind: 'image' };
  if (s.stage === 'loss') return { kind: 'loss' };
  if (s.stage === 'dense') return { kind: 'output', c: s.c ?? tr.target };
  if (s.phase === 'forward') return s.stage === 'conv' ? { kind: 'map', k: s.k!, p: strongest(tr.a[s.k!]) } : { kind: 'pool', k: s.k!, p: strongest(tr.pooled[s.k!]) };
  if (s.phase === 'backward' && s.stage === 'pool') return { kind: 'map', k: s.k!, p: strongest(tr.dA[s.k!]) };
  return { kind: 'filter', k: s.k! };
}

export function CnnView({ frame, pixels, selection, onSelect, training, classes }: { frame: CnnFrame; pixels: number[]; selection: CnnSelection; onSelect: (s: CnnSelection) => void; training: boolean; classes: string[] }) {
  const { t } = useI18n();
  const T = t.cnn;
  const zoom = useZoom();
  const tr = frame.trace, K = frame.model.config.filters;
  const progress = cnnProgress(frame, training);
  const phase = training ? 'train' : frame.plan[frame.cursor].phase;
  const showGradients = phase === 'backward';
  const filters = Array.from({ length: K }, (_, k) => frame.model.params[`f${k}`]);
  const filterScale = Math.max(0.3, ...filters.flatMap(f => f.slice(0, 9).map(Math.abs)));
  const mapScale = Math.max(0.3, ...tr.a.flat().map(Math.abs));
  const gradScale = (values: number[]) => Math.max(1e-6, ...values.map(Math.abs));
  const dAScale = gradScale(tr.dZ.flat()), dFlatScale = gradScale(tr.dFlat), filterGradScale = gradScale(filters.flatMap((_, k) => tr.grads[`f${k}`].slice(0, 9)));
  const field = selection.kind === 'map' ? new Set(receptiveField(Math.floor(selection.p / MAP), selection.p % MAP)) : null;
  const window = selection.kind === 'pool' ? new Set(poolWindow(Math.floor(selection.p / POOLED), selection.p % POOLED)) : null;
  const style = { '--cell': `${1.4 * zoom.zoom}rem` } as CSSProperties;
  const isSel = (kind: string, k?: number, p?: number) => selection.kind === kind && (k === undefined || ('k' in selection && selection.k === k)) && (p === undefined || ('p' in selection && selection.p === p));
  return <div className="network-stage" data-tour="network">
    <div className="network-scroll">
      <div className={`cnn-pipeline phase-${phase}`} style={style}>
        <div className="cnn-column">
          <ColumnHeader title={T.columns.input} hint={T.columnHints.input}/>
          <button className={`pixel-grid ${isSel('image') ? 'selected' : ''}`} style={{ gridTemplateColumns: `repeat(${IMAGE}, var(--cell))` }} onClick={() => onSelect({ kind: 'image' })} aria-label={T.inputAria}>
            {pixels.map((v, i) => <span key={i} className={`cell ${field?.has(i) ? 'in-field' : ''}`} style={{ background: `color-mix(in srgb, #2d2542 ${Math.round(v * 100)}%, white)` }} title={T.pixelAria(Math.floor(i / IMAGE) + 1, i % IMAGE + 1, format(v, 2))}/>)}
          </button>
        </div>
        <span className="cnn-arrow" aria-hidden>→</span>
        <div className="cnn-rows">
          <div className="cnn-row cnn-row-header">
            <ColumnHeader title={T.columns.filters} hint={showGradients ? '∂L/∂w' : T.columnHints.filters}/>
            <span/>
            <ColumnHeader title={T.columns.maps} hint={showGradients ? '∂L/∂z' : T.columnHints.maps}/>
            <span/>
            <ColumnHeader title={T.columns.pooled} hint={showGradients ? '∂L/∂p' : T.columnHints.pooled}/>
          </div>
          {Array.from({ length: K }, (_, k) => {
            const convShown = progress.conv(k), poolShown = progress.pool(k);
            const filterGrad = showGradients && progress.backConv(k);
            const mapGrad = showGradients && progress.backPool(k);
            return <div className="cnn-row" key={k}>
              <button className={`kernel-grid ${isSel('filter', k) || (selection.kind === 'map' && selection.k === k) ? 'selected' : ''}`} style={{ gridTemplateColumns: `repeat(${KERNEL}, calc(var(--cell) * 1.35))` }} onClick={() => onSelect({ kind: 'filter', k })} aria-label={T.filterAria(k + 1)}>
                {filters[k].slice(0, 9).map((w, q) => { const v = filterGrad ? tr.grads[`f${k}`][q] : w, s = filterGrad ? filterGradScale : filterScale; return <span key={q} className="cell" style={{ background: valueColor(v, s), color: inkColor(v, s) }} title={format(v)}>{format(v, 1)}</span>; })}
              </button>
              <span className="cnn-arrow small" aria-hidden>→</span>
              <div className="map-grid" style={{ gridTemplateColumns: `repeat(${MAP}, var(--cell))` }}>
                {tr.a[k].map((v, p) => { const g = tr.dZ[k][p]; const value = mapGrad ? g : v; return <button key={p} className={`cell ${isSel('map', k, p) ? 'selected' : ''} ${selection.kind === 'pool' && selection.k === k && window?.has(p) ? 'in-field' : ''} ${selection.kind === 'pool' && selection.k === k && tr.winners[k].includes(p) && frame.model.config.pooling === 'max' && window?.has(p) ? 'winner' : ''}`} style={convShown ? { background: valueColor(value, mapGrad ? dAScale : mapScale) } : undefined} onClick={() => onSelect({ kind: 'map', k, p })} aria-label={T.mapAria(k + 1, Math.floor(p / MAP) + 1, p % MAP + 1, convShown ? format(value) : null)}/>; })}
              </div>
              <span className="cnn-arrow small" aria-hidden>→</span>
              <div className="pool-grid" style={{ gridTemplateColumns: `repeat(${POOLED}, calc(var(--cell) * 1.45))` }}>
                {tr.pooled[k].map((v, p) => { const value = mapGrad ? tr.dFlat[k * 4 + p] : v; return <button key={p} className={`cell ${isSel('pool', k, p) ? 'selected' : ''}`} style={poolShown ? { background: valueColor(value, mapGrad ? dFlatScale : mapScale), color: inkColor(value, mapGrad ? dFlatScale : mapScale) } : undefined} onClick={() => onSelect({ kind: 'pool', k, p })} aria-label={T.poolAria(k + 1, Math.floor(p / POOLED) + 1, p % POOLED + 1, poolShown ? format(value) : null)}>{poolShown ? format(value, 1) : '—'}</button>; })}
              </div>
            </div>;
          })}
        </div>
        <span className="cnn-arrow" aria-hidden>→</span>
        <div className="cnn-column output-column">
          <ColumnHeader title={T.columns.output} hint={T.columnHints.output}/>
          {Array.from({ length: CLASSES }, (_, c) => {
            const p = tr.probs[c], shown = progress.dense;
            return <button key={c} className={`class-bar ${isSel('output') && selection.kind === 'output' && selection.c === c ? 'selected' : ''} ${tr.target === c ? 'is-target' : ''}`} onClick={() => onSelect({ kind: 'output', c })} aria-label={T.outputAria(classes[c], shown ? `${Math.round(p * 100)}%` : null)}>
              <span className="class-name">{classes[c]}{tr.target === c && <small className="target-chip">{t.common.target}</small>}</span>
              <span className="bar-track"><span className="bar-fill" style={{ width: shown ? `${p * 100}%` : 0 }}/></span>
              <span className="bar-value">{shown ? `${Math.round(p * 100)}%` : '—'}</span>
            </button>;
          })}
        </div>
      </div>
    </div>
    <div className="canvas-footer"><span>{T.legend}</span><span className="canvas-hint">{t.common.clickToInspect}</span><ZoomControls zoom={zoom}/></div>
  </div>;
}
function ColumnHeader({ title, hint }: { title: string; hint: string }) { return <div className="column-header"><span className="layer-label">{title}</span><span className="layer-sublabel">{hint}</span></div>; }
