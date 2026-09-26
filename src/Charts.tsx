import { useMemo } from 'react';
import { forward, format, type Frame, type Problem, type Sample } from './engine';

export function LossChart({ frame }: { frame: Frame }) {
  const history = frame.history;
  const max = Math.max(0.01, ...history.map(m => m.loss)) * 1.15;
  const points = history.map((m, i) => `${40 + i * 460 / Math.max(history.length - 1, 1)},${125 - m.loss / max * 102}`).join(' ');
  return <div className="loss-plot"><svg viewBox="0 0 530 163" role="img" aria-label={`Mean training loss ${format(frame.metric.loss)} after ${format(frame.metric.epoch, 2)} epochs`}>
    {[0, 0.5, 1].map(r => <g key={r}><line x1="40" y1={125 - r * 102} x2="505" y2={125 - r * 102} stroke="#e9eaf0" strokeDasharray="3 4"/><text x="30" y={129 - r * 102} className="chart-label" textAnchor="end">{(max * r).toFixed(2)}</text></g>)}
    {history.length > 1 && <polygon points={`40,125 ${points} 500,125`} fill="url(#lossFill)"/>}<defs><linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#8f7ede" stopOpacity="0.2"/><stop offset="100%" stopColor="#8f7ede" stopOpacity="0"/></linearGradient></defs>
    <polyline points={points} fill="none" stroke="#8470d5" strokeWidth="2.5" strokeLinejoin="round"/>
    {history.length === 1 && <circle cx="40" cy={125 - history[0].loss / max * 102} r="4" fill="#8470d5"/>}
    <text x="40" y="149" className="chart-label">{format(history[0].epoch, 0)}</text><text x="500" y="149" className="chart-label" textAnchor="end">{format(history.at(-1)!.epoch, 1)}</text><text x="270" y="149" className="chart-label" textAnchor="middle">Epochs</text>
  </svg>{history.length === 1 && <span className="chart-empty">Every little improvement starts here.</span>}</div>;
}
export function DataView({ frame, data, problem, onSample }: { frame: Frame; data: Sample[]; problem: Problem; onSample: (index: number) => void }) {
  const predictions = useMemo(() => data.map(s => forward(frame.network, s.x).a.at(-1)!), [frame.network, data]);
  const grid = useMemo(() => {
    if (problem === 'xor') return [];
    return Array.from({ length: 24 * 24 }, (_, k) => {
      const output = forward(frame.network, [-1 + (k % 24 + 0.5) / 12, 1 - (Math.floor(k / 24) + 0.5) / 12]).a.at(-1)!;
      const prediction = output.length === 2 ? Number(output[1] > output[0]) : Number(output[0] >= 0.5);
      return <rect key={k} x={28 + k % 24 * 7} y={7 + Math.floor(k / 24) * 7} width="7.2" height="7.2" fill={prediction ? '#e7e0fa' : '#deefe8'}/>;
    });
  }, [frame.network, problem]);
  if (problem === 'xor') return <table className="data-table"><thead><tr><th>x₁</th><th>x₂</th><th>Target</th><th>Prediction</th><th aria-label="Correctness"/></tr></thead><tbody>{data.map((sample, index) => {
    const output = predictions[index], value = output.length === 2 ? output[1] : output[0];
    const correct = (output.length === 2 ? Number(output[1] > output[0]) : Number(value >= 0.5)) === sample.y;
    return <tr key={index} className={frame.sampleIndex === index ? 'current-sample' : ''}><td><button aria-label={`Inspect XOR sample ${sample.x.join(', ')}`} onClick={() => onSample(index)}>{sample.x[0]}</button></td><td>{sample.x[1]}</td><td><span className={`target-value class-${sample.y}`}>{sample.y}</span></td><td className="mono">{format(value, 3)}</td><td><span className={`prediction-dot ${correct ? 'correct' : 'incorrect'}`} title={correct ? 'Correct class' : 'Incorrect class'}/></td></tr>;
  })}</tbody></table>;
  return <div className="scatter-content"><svg viewBox="0 0 215 197" className="scatter" role="group" aria-label="2D training points and decision regions">{grid}<line x1="112" x2="112" y1="7" y2="175" stroke="#fff"/><line x1="28" x2="196" y1="91" y2="91" stroke="#fff"/>{data.map((sample, i) => <circle role="button" tabIndex={0} aria-label={`Inspect point ${i + 1}, class ${sample.y}`} key={i} cx={28 + (sample.x[0] + 1) * 84} cy={7 + (1 - sample.x[1]) * 84} r={frame.sampleIndex === i ? 5 : 3} fill={sample.y ? '#8d76cf' : '#4c9b85'} stroke={frame.sampleIndex === i ? '#292337' : '#fff'} strokeWidth={frame.sampleIndex === i ? 2 : 0.8} onClick={() => onSample(i)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSample(i); } }}/>) }<text x="28" y="190" className="chart-label">−1</text><text x="187" y="190" className="chart-label">+1</text><text x="108" y="191" className="chart-label">x₁</text><text x="11" y="93" className="chart-label">x₂</text></svg><div className="scatter-caption"><span><span className="tiny-dot green"/> Class 0</span><span><span className="tiny-dot purple"/> Class 1</span><p>Dots are targets.<br/>Shading is the network's predicted class.</p><small>Click a point to inspect it.</small></div></div>;
}
