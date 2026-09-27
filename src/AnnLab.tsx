import { useRef, useState } from 'react';
import { Check, FlaskConical, MousePointer2, Plus, Trash2 } from 'lucide-react';
import { ACTIVATIONS, DEFAULT_SPECS, RECOMMENDED_RATE, createFrame, createNetwork, dataset, format, parameterCount, step, trainEpoch, type Activation, type LayerSpec, type Network, type Problem } from './engine';
import { NetworkView, type Selection } from './NetworkView';
import { Inspector } from './Inspector';
import { DataView } from './Charts';
import { useI18n } from './i18n';
import { useTimeline } from './useTimeline';
import { Dialog, LabLayout, LearningSettings, NetworkHeading, PhaseTrack, PlaybackBar, ProblemHeader, ResultsCard, Section, SelectField, StepGuide, Stepper, useFrameSelection, useSaved, type LabProps } from './ui';

const PROBLEMS: Problem[] = ['xor', 'circle', 'diagonal'];
const STORAGE_KEY = 'neuron-lab-experiment';

export function AnnLab({ active, picker, notify }: LabProps) {
  const { t } = useI18n();
  const T = t.ann;
  const [specs, setSpecs] = useState<LayerSpec[]>(DEFAULT_SPECS);
  const [problem, setProblem] = useState<Problem>('xor');
  const [seed, setSeed] = useState(42);
  const [rate, setRate] = useState(RECOMMENDED_RATE.xor);
  const data = useRef(dataset('xor'));
  const controls = useTimeline(() => createFrame(createNetwork(DEFAULT_SPECS), data.current), { step: (f, r) => step(f, data.current, r), epoch: (f, r) => trainEpoch(f, data.current, r) }, rate, active);
  const { frame, mode } = controls;
  const [selection, setSelection] = useFrameSelection<Selection>(frame);
  const [editing, setEditing] = useState<Selection | null>(null);
  const saved = useSaved(STORAGE_KEY);
  const event = frame.plan[frame.cursor];

  function reset(nextSpecs = specs, nextProblem = problem, nextSeed = seed, network?: Network) {
    data.current = dataset(nextProblem);
    controls.reset(createFrame(network ?? createNetwork(nextSpecs, nextSeed), data.current));
    setSpecs(nextSpecs); setProblem(nextProblem); setSeed(nextSeed);
  }
  function updateLayer(index: number, changes: Partial<LayerSpec>) {
    const updated = specs.map((s, i) => i === index ? { ...s, ...changes } : s);
    if (index === specs.length - 1 && changes.activation === 'softmax') updated[index].size = 2;
    reset(updated);
  }
  function chooseSample(index: number) {
    controls.push(f => createFrame(f.cursor === f.plan.length - 1 ? f.network : f.trace.source, data.current, index, f.samplesSeen, f.history));
    notify(t.common.sampleToast);
  }
  function save() {
    try { saved.save({ version: 1, specs, problem, seed, rate, network: frame.network }); notify(t.common.savedToast); }
    catch { notify(t.common.storageUnavailable); }
  }
  function restore() {
    try {
      const s = saved.load() as { version: number; specs: LayerSpec[]; problem: Problem; seed: number; rate: number; network: Network } | null;
      if (!s || s.version !== 1 || !PROBLEMS.includes(s.problem) || !Array.isArray(s.specs) || !Array.isArray(s.network?.layers)) throw new Error();
      reset(s.specs, s.problem, s.seed, s.network); setRate(s.rate); notify(t.common.restoredToast);
    } catch { notify(t.common.restoreFailed); }
  }
  const sample = data.current[frame.sampleIndex];
  const S = T.step;
  const [title, text] = mode === 'train' ? [S.trainTitle, S.trainText(data.current.length)]
    : event.phase === 'input' ? [S.inputTitle, S.inputText(sample.x.map(v => format(v, 2)).join(', '), sample.y)]
    : event.phase === 'forward' ? [S.forwardTitle(event.layer! + 1, event.neuron! + 1), S.forwardText]
    : event.phase === 'loss' ? [S.lossTitle, S.lossText(format(frame.trace.loss, 6))]
    : event.phase === 'backward' ? [S.backwardTitle(event.layer! + 1, event.neuron! + 1), S.backwardText]
    : [S.updateTitle(event.bias ? S.bias : S.weight(event.input! + 1), event.layer! + 1, event.neuron! + 1), S.updateText];
  const output = specs.at(-1)!;
  const sidebar = <>
    <Section number={2} label={t.sections.problem} htmlFor="ann-problem" tour="problem">
      <SelectField id="ann-problem" value={problem} onChange={v => { reset(specs, v as Problem); setRate(RECOMMENDED_RATE[v as Problem]); }} icon={<FlaskConical size={17}/>}>{PROBLEMS.map(p => <option key={p} value={p}>{T.problems[p].option}</option>)}</SelectField>
      <p className="field-help">{T.problems[problem].help}</p>
    </Section>
    <Section number={3} label={t.sections.shape} className="architecture-section">
      <div className="architecture-heading"><span>{T.architecture}</span><span className="mono">2 → {specs.map(s => s.size).join(' → ')}</span></div>
      <div className="layer-row fixed-layer"><span className="layer-dot input-dot"/><div><strong>{T.inputLayer}</strong><small>{T.inputFixed}</small></div><span className="neuron-count">2</span></div>
      <div className="layers-stack">{specs.slice(0, -1).map((spec, i) => <div className="layer-editor" key={i}>
        <div className="layer-row"><span className="layer-dot"/><strong>{T.hidden(i + 1)}</strong><Stepper value={spec.size} min={1} max={12} onChange={size => updateLayer(i, { size })} decrease={T.removeNeuron(i + 1)} increase={T.addNeuron(i + 1)}/><button className="icon-button delete-layer" aria-label={T.removeLayer(i + 1)} title={T.removeLayer(i + 1)} onClick={() => reset(specs.filter((_, k) => k !== i))}><Trash2 size={15}/></button></div>
        <select className="activation-select" aria-label={T.hiddenActivation(i + 1)} value={spec.activation} onChange={e => updateLayer(i, { activation: e.target.value as Activation })}>{ACTIVATIONS.map(a => <option key={a}>{a}</option>)}</select>
      </div>)}</div>
      <button className="add-layer" disabled={specs.length >= 7} onClick={() => reset([...specs.slice(0, -1), { size: 3, activation: 'tanh' }, output])}><Plus size={15}/> {T.addLayer}</button>
      <div className="output-editor">
        <div className="layer-row"><span className="layer-dot output-dot"/><strong>{T.outputLayer}</strong><select aria-label={T.outputNeurons} value={output.size} onChange={e => updateLayer(specs.length - 1, { size: Number(e.target.value) })}><option value="1" disabled={output.activation === 'softmax'}>{T.oneNeuron}</option><option value="2">{T.twoNeurons}</option></select></div>
        <select className="activation-select" aria-label={T.outputActivation} value={output.activation} onChange={e => updateLayer(specs.length - 1, { activation: e.target.value as Activation })}>{ACTIVATIONS.map(a => <option key={a}>{a}</option>)}</select>
      </div>
      <p className="field-help">{t.common.architectureResets} {T.limits}</p>
    </Section>
    <LearningSettings number={4} rate={rate} onRate={v => { setRate(v); controls.clearFuture(); }} locked={controls.lockedRate || controls.playing} seed={seed} onSeed={v => reset(specs, problem, v)} onRecommended={() => { reset(DEFAULT_SPECS, problem, 42); setRate(RECOMMENDED_RATE[problem]); notify(t.common.recommendedToast); }}/>
  </>;
  return <LabLayout id="ann" active={active} picker={picker} sidebar={sidebar}>
    <ProblemHeader title={T.problems[problem].title} description={T.problems[problem].description} mode={mode} onMode={controls.changeMode}/>
    <div className="workbench">
      <section className="network-card card">
        <NetworkHeading title={T.networkTitle} stats={[t.common.layers(specs.length + 1), t.common.neurons(2 + specs.reduce((s, l) => s + l.size, 0)), t.common.parameters(parameterCount(frame.network))]} onReset={() => reset()} onSave={save} onRestore={restore} hasSaved={saved.hasSaved}/>
        <PhaseTrack phase={event.phase} mode={mode}/>
        <NetworkView frame={frame} selection={selection} onSelect={s => { setSelection(s); controls.setPlaying(false); }} training={mode === 'train'}/>
        <StepGuide phase={event.phase} title={title} text={text} frame={frame} mode={mode}/>
        <PlaybackBar controls={controls}/>
      </section>
      <Inspector frame={frame} selection={selection} rate={frame.learningRate ?? rate} onEdit={s => { controls.setPlaying(false); setEditing(s); }}/>
    </div>
    {controls.timeline.error && <div className="error-banner" role="alert">{t.common.diverged}<button onClick={() => reset()}>{t.common.resetNetwork}</button></div>}
    <div className="results-grid" data-tour="results">
      <section className="card data-card"><div className="panel-heading"><h2>{problem === 'xor' ? T.data.table : T.data.boundary}</h2><span className="small-chip">{t.common.examples(data.current.length)}</span></div><DataView frame={frame} data={data.current} problem={problem} onSample={chooseSample}/><div className="data-footer"><MousePointer2 size={14}/>{problem === 'xor' ? T.data.clickXor : T.data.pointsNote}</div></section>
      <ResultsCard frame={frame}/>
    </div>
    <Dialog open={editing !== null} onClose={() => setEditing(null)} label={T.editor.dialog}>
      {editing && <ParameterEditor key={`${editing.layer}-${editing.neuron}`} network={frame.network} selected={editing} onApply={network => { reset(specs, problem, seed, network); setEditing(null); notify(T.editor.toast); }}/>}
    </Dialog>
  </LabLayout>;
}
function ParameterEditor({ network, selected, onApply }: { network: Network; selected: Selection; onApply: (n: Network) => void }) {
  const { t } = useI18n();
  const T = t.ann.editor;
  const { layer: l, neuron: j } = selected;
  const [values, setValues] = useState(network.layers[l].weights[j].map(String));
  const [bias, setBias] = useState(String(network.layers[l].biases[j]));
  return <form onSubmit={e => { e.preventDefault(); const copy = structuredClone(network); copy.layers[l].weights[j] = values.map(Number); copy.layers[l].biases[j] = Number(bias); onApply(copy); }}>
    <span className="eyebrow">{T.eyebrow}</span><h2>{T.title}</h2><p>{T.text(l + 1, j + 1)}</p>
    <div className="parameter-fields">{values.map((value, i) => <label key={i}>{T.weightFrom(i + 1)}<input aria-label={T.weightFrom(i + 1)} type="number" step="any" min="-100" max="100" required value={value} onChange={e => setValues(v => v.map((x, k) => k === i ? e.target.value : x))}/></label>)}<label>{T.bias}<input aria-label={T.bias} type="number" min="-100" max="100" step="any" required value={bias} onChange={e => setBias(e.target.value)}/></label></div>
    <button className="primary-button" type="submit">{T.apply} <Check size={17}/></button>
  </form>;
}
