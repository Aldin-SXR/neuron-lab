import { useMemo } from 'react';
import { forward, type Frame, type Problem, type Sample } from './engine';
import { useI18n } from './i18n';

export function DataView({ frame, data, problem, onSample }: { frame: Frame; data: Sample[]; problem: Problem; onSample: (index: number) => void }) {
  const { t, num } = useI18n();
  const T = t.ann.data;
  const predictions = useMemo(() => data.map(s => forward(frame.network, s.x).a.at(-1)!), [frame.network, data]);
  const grid = useMemo(() => {
    if (problem === 'xor') return [];
    return Array.from({ length: 24 * 24 }, (_, k) => {
      const output = forward(frame.network, [-1 + (k % 24 + 0.5) / 12, 1 - (Math.floor(k / 24) + 0.5) / 12]).a.at(-1)!;
      const prediction = output.length === 2 ? Number(output[1] > output[0]) : Number(output[0] >= 0.5);
      return <rect key={k} x={28 + k % 24 * 7} y={7 + Math.floor(k / 24) * 7} width="7.2" height="7.2" fill={prediction ? '#e4dcfa' : '#d8efe5'}/>;
    });
  }, [frame.network, problem]);
  if (problem === 'xor') return <table className="data-table"><thead><tr><th>x₁</th><th>x₂</th><th>{t.common.target}</th><th>{t.common.prediction}</th><th aria-label={t.common.correct}/></tr></thead><tbody>{data.map((sample, index) => {
    const output = predictions[index], value = output.length === 2 ? output[1] : output[0];
    const correct = (output.length === 2 ? Number(output[1] > output[0]) : Number(value >= 0.5)) === sample.y;
    return <tr key={index} className={frame.sampleIndex === index ? 'current-sample' : ''} onClick={() => onSample(index)}><td><button aria-label={T.inspectXor(sample.x.join(', '))} onClick={e => { e.stopPropagation(); onSample(index); }}>{sample.x[0]}</button></td><td>{sample.x[1]}</td><td><span className={`target-value class-${sample.y}`}>{sample.y}</span></td><td className="mono">{num(value, 3)}</td><td><span className={`prediction-dot ${correct ? 'correct' : 'incorrect'}`} title={correct ? t.common.correct : t.common.incorrect}/></td></tr>;
  })}</tbody></table>;
  return <div className="scatter-content"><svg viewBox="0 0 215 197" className="scatter" role="group" aria-label={T.scatterLabel}>{grid}<line x1="112" x2="112" y1="7" y2="175" stroke="#fff"/><line x1="28" x2="196" y1="91" y2="91" stroke="#fff"/>{data.map((sample, i) => <circle role="button" tabIndex={0} aria-label={T.inspectPoint(i + 1, sample.y)} key={i} cx={28 + (sample.x[0] + 1) * 84} cy={7 + (1 - sample.x[1]) * 84} r={frame.sampleIndex === i ? 5 : 3.2} fill={sample.y ? '#7a62d0' : '#3f9a7f'} stroke={frame.sampleIndex === i ? '#241d33' : '#fff'} strokeWidth={frame.sampleIndex === i ? 2 : 0.8} onClick={() => onSample(i)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSample(i); } }}/>)}<text x="28" y="190" className="chart-label">−1</text><text x="187" y="190" className="chart-label">+1</text><text x="108" y="191" className="chart-label">x₁</text><text x="11" y="93" className="chart-label">x₂</text></svg><div className="scatter-caption"><span><span className="tiny-dot green"/> {T.class0}</span><span><span className="tiny-dot purple"/> {T.class1}</span><p>{T.dots}</p></div></div>;
}
