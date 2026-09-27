import { format, type Frame } from './engine';
import { useI18n } from './i18n';
import { useContext } from 'react';
import { ShellContext, ZoomControls, useZoom } from './ui';

export interface Selection { layer: number; neuron: number; input?: number; bias?: boolean }
export function NetworkView({ frame, selection, onSelect, training }: { frame: Frame; selection: Selection | null; onSelect: (s: Selection) => void; training: boolean }) {
  const { t } = useI18n();
  const zoom = useZoom();
  // The text-size setting enlarges the whole diagram so its labels grow with the rest of the page.
  const scale = zoom.zoom * useContext(ShellContext).textScale;
  const event = frame.plan[frame.cursor];
  const sizes = [2, ...frame.network.layers.map(l => l.size)];
  const width = Math.max(640, sizes.length * 170);
  const height = Math.max(340, Math.max(...sizes) * 78 + 110);
  const position = (l: number, n: number) => ({ x: 72 + l * (width - 144) / (sizes.length - 1), y: height / 2 + (n - (sizes[l] - 1) / 2) * 78 + 8 });
  const active = selection ?? (event.layer !== undefined ? { layer: event.layer, neuron: event.neuron!, input: event.input, bias: event.bias } : null);
  function visible(l: number, j: number) {
    if (l === 0 || training) return true;
    return frame.plan.slice(0, frame.cursor + 1).some(e => e.phase === 'forward' && e.layer === l - 1 && e.neuron === j);
  }
  const keyActivate = (e: React.KeyboardEvent, action: () => void) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); action(); } };
  return <div className="network-stage" data-tour="network">
    <div className="network-scroll">
      <svg className={`network-svg phase-${training ? 'train' : event.phase}`} viewBox={`0 0 ${width} ${height}`} style={{ width: `${zoom.zoom * 100}%`, minWidth: width * scale, height: height * scale }} role="group" aria-label={t.ann.svgLabel}>
        {sizes.map((size, l) => <g key={`label-${l}`}><text x={position(l, 0).x} y="30" className="layer-label">{t.ann.layerLabel(l, l === sizes.length - 1)}</text><text x={position(l, 0).x} y="50" className="layer-sublabel">{l === 0 ? t.ann.features : frame.network.layers[l - 1].activation}</text><text x={position(l, 0).x} y={height - 14} className="layer-sublabel">{t.ann.neuronCount(size)}</text></g>)}
        {frame.network.layers.map((layer, l) => layer.weights.map((row, j) => row.map((weight, i) => {
          const from = position(l, i), to = position(l + 1, j);
          const selected = active?.layer === l && active.neuron === j && (active.input === undefined || active.input === i) && !active.bias;
          const color = weight >= 0 ? '#7a62d0' : '#d39a55';
          return <g key={`${l}-${j}-${i}`} className="connection">
            <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke={color} strokeWidth={selected ? 3 : Math.min(3.5, 0.8 + Math.abs(weight) * 0.7)} opacity={selected ? 0.95 : 0.38} className={selected ? 'active-wire' : ''}/>
            <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke="transparent" strokeWidth="12" className="wire-hit" role="button" tabIndex={0} aria-label={t.ann.weightAria(l + 1, j + 1, i + 1, format(weight))} onClick={() => onSelect({ layer: l, neuron: j, input: i })} onKeyDown={e => keyActivate(e, () => onSelect({ layer: l, neuron: j, input: i }))}><title>{t.ann.weightTitle(format(weight))}</title></line>
          </g>;
        })))}
        {sizes.map((size, l) => Array.from({ length: size }, (_, j) => {
          const p = position(l, j);
          const selected = active?.layer === l - 1 && active.neuron === j;
          const shown = visible(l, j);
          const value = frame.trace.a[l][j];
          return <g key={`node-${l}-${j}`} className={`neuron ${selected ? 'selected' : ''} ${l === 0 ? 'input-neuron' : l === sizes.length - 1 ? 'output-neuron' : ''}`} role="button" tabIndex={0} aria-label={t.ann.neuronAria(l, j + 1, shown ? format(value) : null)} onClick={() => onSelect({ layer: l - 1, neuron: j })} onKeyDown={e => keyActivate(e, () => onSelect({ layer: l - 1, neuron: j }))}>
            {selected && <circle cx={p.x} cy={p.y} r="31" className="selection-ring"/>}
            <circle cx={p.x} cy={p.y} r="25" className="node-circle" style={shown && l > 0 ? { fill: `color-mix(in srgb, ${l === sizes.length - 1 ? '#cdeee0' : '#ddd3fb'} ${Math.min(100, 25 + Math.abs(value) * 60)}%, white)` } : undefined}/>
            <text x={p.x} y={p.y + 5} className="node-value">{shown ? value.toFixed(l === 0 ? 1 : 2) : '—'}</text>
            <text x={p.x} y={p.y + 41} className="node-index">{l === 0 ? `x${j + 1}` : l === sizes.length - 1 ? (size === 2 ? t.ann.classLabel(j) : 'ŷ') : `a${j + 1}`}</text>
          </g>;
        }))}
        {active && active.layer >= 0 && active.input !== undefined && (() => {
          const a = position(active.layer, active.input), b = position(active.layer + 1, active.neuron);
          return <g className="weight-label"><rect x={(a.x + b.x) / 2 - 38} y={(a.y + b.y) / 2 - 14} width="76" height="28" rx="7"/><text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 + 5}>{format(frame.network.layers[active.layer].weights[active.neuron][active.input], 3)}</text></g>;
        })()}
      </svg>
    </div>
    <div className="canvas-footer"><span><span className="tiny-dot purple"/> {t.common.positive} <span className="tiny-dot sand"/> {t.common.negative}</span><span className="canvas-hint">{t.common.clickToInspect}</span><ZoomControls zoom={zoom}/></div>
  </div>;
}
