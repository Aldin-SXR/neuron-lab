import { Minus, Plus, Maximize2 } from 'lucide-react';
import { useState } from 'react';
import { format, type Frame } from './engine';

export interface Selection { layer: number; neuron: number; input?: number; bias?: boolean }
export function NetworkView({ frame, selection, onSelect, training }: { frame: Frame; selection: Selection | null; onSelect: (s: Selection) => void; training: boolean }) {
  const [zoom, setZoom] = useState(1);
  const event = frame.plan[frame.cursor];
  const sizes = [2, ...frame.network.layers.map(l => l.size)];
  const width = Math.max(640, sizes.length * 170);
  const height = Math.max(330, Math.max(...sizes) * 62 + 96);
  const position = (l: number, n: number) => ({ x: 68 + l * (width - 136) / (sizes.length - 1), y: height / 2 + (n - (sizes[l] - 1) / 2) * 62 + 8 });
  const active = selection ?? (event.layer !== undefined ? { layer: event.layer, neuron: event.neuron!, input: event.input, bias: event.bias } : null);
  function visible(l: number, j: number) {
    if (l === 0 || training) return true;
    return frame.plan.slice(0, frame.cursor + 1).some(e => e.phase === 'forward' && e.layer === l - 1 && e.neuron === j);
  }
  return <div className="network-stage">
    <div className="network-scroll">
      <svg className={`network-svg phase-${event.phase}`} viewBox={`0 0 ${width} ${height}`} style={{ width: `${zoom * 100}%`, minWidth: width * zoom, height: height * zoom }} role="group" aria-label="Interactive neural network. Select a neuron or a connection to inspect its calculation.">
        {sizes.map((size, l) => <g key={`label-${l}`}><text x={position(l, 0).x} y="29" className="layer-label">{l === 0 ? 'INPUT' : l === sizes.length - 1 ? 'OUTPUT' : `HIDDEN ${l}`}</text><text x={position(l, 0).x} y="47" className="layer-sublabel">{l === 0 ? '2 features' : frame.network.layers[l - 1].activation}</text><text x={position(l, 0).x} y={height - 15} className="layer-sublabel">{size} {size === 1 ? 'neuron' : 'neurons'}</text></g>)}
        {frame.network.layers.map((layer, l) => layer.weights.map((row, j) => row.map((weight, i) => {
          const from = position(l, i), to = position(l + 1, j);
          const selected = active?.layer === l && active.neuron === j && (active.input === undefined || active.input === i) && !active.bias;
          const color = weight >= 0 ? '#7a73c4' : '#d0a57a';
          return <g key={`${l}-${j}-${i}`} className="connection">
            <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke={color} strokeWidth={selected ? 2.5 : Math.min(3, 0.7 + Math.abs(weight) * 0.65)} opacity={selected ? 0.95 : 0.3} className={selected ? 'active-wire' : ''}/>
            <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke="transparent" strokeWidth="12" className="wire-hit" role="button" tabIndex={0} aria-label={`Weight layer ${l + 1} neuron ${j + 1} input ${i + 1}: ${format(weight)}`} onClick={() => onSelect({ layer: l, neuron: j, input: i })} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect({ layer: l, neuron: j, input: i }); } }}><title>{`w = ${format(weight)} · click to inspect`}</title></line>
          </g>;
        })))}
        {sizes.map((size, l) => Array.from({ length: size }, (_, j) => {
          const p = position(l, j);
          const selected = active?.layer === l - 1 && active.neuron === j;
          const shown = visible(l, j);
          const value = frame.trace.a[l][j];
          return <g key={`node-${l}-${j}`} className={`neuron ${selected ? 'selected' : ''} ${l === 0 ? 'input-neuron' : l === sizes.length - 1 ? 'output-neuron' : ''}`} role="button" tabIndex={0} aria-label={`${l === 0 ? 'Input' : `Layer ${l} neuron`} ${j + 1}${shown ? ` value ${format(value)}` : ', not calculated yet'}`} onClick={() => onSelect({ layer: l - 1, neuron: j })} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect({ layer: l - 1, neuron: j }); } }}>
            {selected && <circle cx={p.x} cy={p.y} r="29" className="selection-ring"/>}
            <circle cx={p.x} cy={p.y} r="23" className="node-circle" style={shown && l > 0 ? { fill: `color-mix(in srgb, ${l === sizes.length - 1 ? '#ddf2e9' : '#ddd5fc'} ${Math.min(100, 25 + Math.abs(value) * 60)}%, white)` } : undefined}/>
            <text x={p.x} y={p.y + 4} className="node-value">{shown ? value.toFixed(l === 0 ? 1 : 2) : '—'}</text>
            <text x={p.x} y={p.y + 38} className="node-index">{l === 0 ? `x${j + 1}` : l === sizes.length - 1 ? (size === 2 ? `class ${j}` : 'ŷ') : `a${j + 1}`}</text>
          </g>;
        }))}
        {active && active.layer >= 0 && active.input !== undefined && (() => {
          const a = position(active.layer, active.input), b = position(active.layer + 1, active.neuron);
          return <g className="weight-label"><rect x={(a.x + b.x) / 2 - 34} y={(a.y + b.y) / 2 - 12} width="68" height="24" rx="6"/><text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 + 4}>{format(frame.network.layers[active.layer].weights[active.neuron][active.input], 3)}</text></g>;
        })()}
      </svg>
    </div>
    <div className="canvas-footer"><span><span className="tiny-dot purple"/> Positive weight <span className="tiny-dot sand"/> Negative weight</span><div className="zoom-controls"><button title="Zoom out" aria-label="Zoom out" onClick={() => setZoom(z => Math.max(0.65, z - 0.15))}><Minus size={14}/></button><span>{Math.round(zoom * 100)}%</span><button title="Zoom in" aria-label="Zoom in" onClick={() => setZoom(z => Math.min(1.8, z + 0.15))}><Plus size={14}/></button><button title="Reset zoom" aria-label="Reset zoom" onClick={() => setZoom(1)}><Maximize2 size={13}/></button></div></div>
  </div>;
}
