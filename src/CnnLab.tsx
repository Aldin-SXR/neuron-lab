import { useMemo, useRef, useState } from 'react';
import { Eraser, FlaskConical, MousePointer2 } from 'lucide-react';
import { CLASSES, DEFAULT_CNN, IMAGE, cnnDataset, cnnForward, cnnLab, createCnn, type CnnConfig, type CnnFrame, type CnnImage, type CnnModel, type CnnProblem, type ConvActivation, type Pooling } from './cnn';
import { CnnView, defaultCnnSelection, type CnnSelection } from './CnnView';
import { CnnInspector } from './CnnInspector';
import { argmax } from './lab';
import { useI18n } from './i18n';
import { useTimeline } from './useTimeline';
import { LabLayout, LearningSettings, NetworkHeading, PhaseTrack, PlaybackBar, ProblemHeader, ResultsCard, Section, SelectField, StepGuide, Stepper, useFrameSelection, useSaved, type LabProps } from './ui';

const PROBLEMS: CnnProblem[] = ['lines', 'shapes'];
const RATE = 0.1, SEED = 7, STORAGE_KEY = 'neuron-lab-cnn';

export function CnnLab({ active, picker, notify }: LabProps) {
  const { t } = useI18n();
  const T = t.cnn;
  const [problem, setProblem] = useState<CnnProblem>('lines');
  const [config, setConfig] = useState<CnnConfig>(DEFAULT_CNN);
  const [seed, setSeed] = useState(SEED);
  const [rate, setRate] = useState(RATE);
  const data = useRef(cnnDataset('lines'));
  const controls = useTimeline(() => cnnLab.createFrame(createCnn(DEFAULT_CNN, SEED), data.current), { step: (f, r) => cnnLab.step(f, data.current, r), epoch: (f, r) => cnnLab.trainEpoch(f, data.current, r) }, rate, active);
  const { frame, mode } = controls;
  const [picked, setPicked] = useFrameSelection<CnnSelection>(frame);
  const selection = picked ?? defaultCnnSelection(frame);
  const saved = useSaved(STORAGE_KEY);
  const classes = T.problems[problem].classes;
  const sample = data.current[frame.sampleIndex];
  const step = frame.plan[frame.cursor];

  function reset(nextConfig = config, nextProblem = problem, nextSeed = seed, model?: CnnModel) {
    data.current = cnnDataset(nextProblem);
    controls.reset(cnnLab.createFrame(model ?? createCnn(nextConfig, nextSeed), data.current));
    setConfig(nextConfig); setProblem(nextProblem); setSeed(nextSeed);
  }
  function chooseSample(index: number) {
    controls.push(f => cnnLab.createFrame(f.cursor === f.plan.length - 1 ? f.model : f.source, data.current, index, f.samplesSeen, f.history));
    notify(t.common.sampleToast);
  }
  function save() {
    try { saved.save({ version: 1, problem, config, seed, rate, model: frame.model }); notify(t.common.savedToast); }
    catch { notify(t.common.storageUnavailable); }
  }
  function restore() {
    try {
      const s = saved.load() as { version: number; problem: CnnProblem; config: CnnConfig; seed: number; rate: number; model: CnnModel } | null;
      if (!s || s.version !== 1 || !PROBLEMS.includes(s.problem) || typeof s.model?.params !== 'object') throw new Error();
      reset(s.model.config, s.problem, s.seed, s.model); setRate(s.rate); notify(t.common.restoredToast);
    } catch { notify(t.common.restoreFailed); }
  }
  const S = T.step, k = (step.k ?? 0) + 1;
  const [title, text] = mode === 'train' ? [S.trainTitle, S.trainText(data.current.length)]
    : step.stage === 'input' ? [S.inputTitle, S.inputText(classes[sample.y])]
    : step.stage === 'loss' ? [S.lossTitle, S.lossText(`${(frame.trace.probs[sample.y] * 100).toFixed(1)}%`, frame.trace.loss.toFixed(4))]
    : step.phase === 'forward' ? step.stage === 'conv' ? [S.convTitle(k), S.convText(config.activation)] : step.stage === 'pool' ? [S.poolTitle(k), config.pooling === 'max' ? S.poolMax : S.poolAverage] : [S.denseTitle, S.denseText(config.filters * 4)]
    : step.phase === 'backward' ? step.stage === 'dense' ? [S.backDenseTitle, S.backDenseText] : step.stage === 'pool' ? [S.backPoolTitle(k), config.pooling === 'max' ? S.backPoolMax : S.backPoolAverage] : [S.backConvTitle(k), S.backConvText]
    : step.stage === 'dense' ? [S.updateDenseTitle(classes[step.c!]), S.updateDenseText(config.filters * 4)] : [S.updateConvTitle(k), S.updateConvText];
  const sidebar = <>
    <Section number={2} label={t.sections.problem} htmlFor="cnn-problem" tour="problem">
      <SelectField id="cnn-problem" value={problem} onChange={v => reset(config, v as CnnProblem)} icon={<FlaskConical size={17}/>}>{PROBLEMS.map(p => <option key={p} value={p}>{T.problems[p].option}</option>)}</SelectField>
      <p className="field-help">{T.problems[problem].help}</p>
    </Section>
    <Section number={3} label={t.sections.shape}>
      <div className="architecture-heading"><span className="mono pipeline-text">{T.pipeline(config.filters)}</span></div>
      <div className="layer-editor"><div className="layer-row"><span className="layer-dot"/><strong>{T.filters}</strong><Stepper value={config.filters} min={1} max={4} onChange={filters => reset({ ...config, filters })} decrease={T.removeFilter} increase={T.addFilter}/></div><p className="field-help">{T.filtersHelp}</p></div>
      <div className="field-grid">
        <label>{T.activation}<select className="activation-select" value={config.activation} onChange={e => reset({ ...config, activation: e.target.value as ConvActivation })}><option value="relu">relu</option><option value="tanh">tanh</option></select></label>
        <label>{T.pooling}<select className="activation-select" value={config.pooling} onChange={e => reset({ ...config, pooling: e.target.value as Pooling })}><option value="max">{T.poolingMax}</option><option value="average">{T.poolingAverage}</option></select></label>
      </div>
      <p className="field-help">{t.common.architectureResets}</p>
    </Section>
    <LearningSettings number={4} rate={rate} onRate={v => { setRate(v); controls.clearFuture(); }} locked={controls.lockedRate || controls.playing} seed={seed} onSeed={v => reset(config, problem, v)} onRecommended={() => { reset(DEFAULT_CNN, problem, SEED); setRate(RATE); notify(t.common.recommendedToast); }}/>
  </>;
  return <LabLayout id="cnn" active={active} picker={picker} sidebar={sidebar}>
    <ProblemHeader title={T.problems[problem].title} description={T.problems[problem].description} mode={mode} onMode={controls.changeMode}/>
    <div className="workbench">
      <section className="network-card card">
        <NetworkHeading title={T.networkTitle} stats={[`${config.filters} × 3×3`, config.activation, config.pooling === 'max' ? T.poolingMax : T.poolingAverage, t.common.parameters(cnnLab.parameterCount(frame.model))]} onReset={() => reset()} onSave={save} onRestore={restore} hasSaved={saved.hasSaved}/>
        <PhaseTrack phase={step.phase} mode={mode}/>
        <CnnView frame={frame} pixels={sample.pixels} selection={selection} onSelect={s => { setPicked(s); controls.setPlaying(false); }} training={mode === 'train'} classes={classes}/>
        <StepGuide phase={step.phase} title={title} text={text} frame={frame} mode={mode}/>
        <PlaybackBar controls={controls}/>
      </section>
      <CnnInspector frame={frame} selection={selection} pixels={sample.pixels} rate={frame.learningRate ?? rate} classes={classes}/>
    </div>
    {controls.timeline.error && <div className="error-banner" role="alert">{t.common.diverged}<button onClick={() => reset()}>{t.common.resetNetwork}</button></div>}
    <div className="results-grid" data-tour="results">
      <ImageData frame={frame} data={data.current} classes={classes} onSample={chooseSample}/>
      <ResultsCard frame={frame}/>
    </div>
  </LabLayout>;
}
function Thumbnail({ pixels }: { pixels: number[] }) {
  return <svg viewBox={`0 0 ${IMAGE} ${IMAGE}`} className="thumb" aria-hidden shapeRendering="crispEdges">{pixels.map((v, i) => <rect key={i} x={i % IMAGE} y={Math.floor(i / IMAGE)} width="1" height="1" fill={`color-mix(in srgb, #2d2542 ${Math.round(v * 100)}%, white)`}/>)}</svg>;
}
function ImageData({ frame, data, classes, onSample }: { frame: CnnFrame; data: CnnImage[]; classes: string[]; onSample: (i: number) => void }) {
  const { t } = useI18n();
  const T = t.cnn.data;
  const [tab, setTab] = useState<'gallery' | 'draw'>('gallery');
  const [drawing, setDrawing] = useState<number[]>(() => Array(IMAGE * IMAGE).fill(0));
  const predictions = useMemo(() => data.map(s => argmax(cnnForward(frame.model, s.pixels).probs)), [frame.model, data]);
  const drawn = useMemo(() => cnnForward(frame.model, drawing).probs, [frame.model, drawing]);
  return <section className="card data-card">
    <div className="panel-heading"><h2>{T.title}</h2><div className="tabs" role="tablist"><button role="tab" aria-selected={tab === 'gallery'} className={tab === 'gallery' ? 'active' : ''} onClick={() => setTab('gallery')}>{T.gallery}</button><button role="tab" aria-selected={tab === 'draw'} className={tab === 'draw' ? 'active' : ''} onClick={() => setTab('draw')}>{T.draw}</button></div></div>
    {tab === 'gallery' ? <>
      <div className="gallery">{data.map((s, i) => <button key={i} className={`thumb-button ${predictions[i] === s.y ? 'correct' : 'incorrect'} ${frame.sampleIndex === i ? 'current' : ''}`} onClick={() => onSample(i)} aria-label={T.inspect(i + 1, classes[s.y])} title={`${classes[s.y]} → ${classes[predictions[i]]}`}><Thumbnail pixels={s.pixels}/><small>{classes[s.y]}</small></button>)}</div>
      <div className="data-footer"><MousePointer2 size={14}/>{T.galleryNote}</div>
    </> : <div className="draw-pad">
      <div className="draw-grid" role="group" aria-label={T.draw}>{drawing.map((v, i) => <button key={i} className={v ? 'on' : ''} aria-pressed={v > 0} aria-label={T.pixelToggle(Math.floor(i / IMAGE) + 1, i % IMAGE + 1)} onClick={() => setDrawing(d => d.map((x, k) => k === i ? 1 - Math.round(x) : x))}/>)}</div>
      <div className="draw-result">
        <p className="field-help">{T.drawNote}</p>
        <strong>{T.predicted}:</strong>
        {Array.from({ length: CLASSES }, (_, c) => <div key={c} className={`class-bar static ${argmax(drawn) === c ? 'is-target' : ''}`}><span className="class-name">{classes[c]}</span><span className="bar-track"><span className="bar-fill" style={{ width: `${drawn[c] * 100}%` }}/></span><span className="bar-value">{Math.round(drawn[c] * 100)}%</span></div>)}
        <div className="draw-actions"><button className="ghost-button" onClick={() => setDrawing(Array(IMAGE * IMAGE).fill(0))}><Eraser size={15}/> {T.clear}</button><button className="ghost-button" onClick={() => setDrawing(data[frame.sampleIndex].pixels.map(v => Number(v > 0.5)))}>{T.useAsSample}</button></div>
      </div>
    </div>}
  </section>;
}
