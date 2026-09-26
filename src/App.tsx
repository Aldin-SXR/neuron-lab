import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowRight, BookOpen, Check, ChevronDown, CircleHelp, Cpu, FlaskConical, GitBranch, Layers3, Minus, MousePointer2, Network as NetworkIcon, Pause, Play, Plus, RotateCcw, Save, SkipBack, SkipForward, Sparkles, Trash2, X, Zap } from 'lucide-react';
import { ACTIVATIONS, DEFAULT_SPECS, createFrame, createNetwork, dataset, format, parameterCount, step, trainEpoch, type Activation, type Frame, type LayerSpec, type Network, type Problem } from './engine';
import { NetworkView, type Selection } from './NetworkView';
import { Inspector } from './Inspector';
import { DataView, LossChart } from './Charts';

type Mode = 'learn' | 'train';
interface Timeline { past: Frame[]; present: Frame; future: Frame[]; error?: string }
type Modal = 'guide' | 'roadmap' | 'edit' | null;
const PROBLEMS: Record<Problem, { title: string; description: string }> = {
  xor: { title: 'The XOR problem', description: 'Different inputs, one output. Can a network learn the difference?' },
  circle: { title: 'Find the inner circle', description: 'Learn a curved boundary that separates two classes of points.' },
  diagonal: { title: 'Draw the dividing line', description: 'Discover a simple boundary between two groups of points.' },
};
const PHASES = [ { id: 'input', name: 'Input', icon: GitBranch }, { id: 'forward', name: 'Forward pass', icon: ArrowRight }, { id: 'loss', name: 'Loss', icon: CircleHelp }, { id: 'backward', name: 'Backpropagation', icon: ArrowDown }, { id: 'update', name: 'Update weights', icon: Zap } ];
function hasSavedExperiment() { try { return Boolean(localStorage.getItem('neuron-lab-experiment')); } catch { return false; } }

export default function App() {
  const [specs, setSpecs] = useState<LayerSpec[]>(DEFAULT_SPECS);
  const [problem, setProblem] = useState<Problem>('xor');
  const [seed, setSeed] = useState(42);
  const [rate, setRate] = useState(0.3);
  const [mode, setMode] = useState<Mode>('learn');
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [timeline, setTimeline] = useState<Timeline>(() => ({ past: [], present: createFrame(createNetwork(DEFAULT_SPECS), dataset('xor')), future: [] }));
  const [modal, setModal] = useState<Modal>(null);
  const [toast, setToast] = useState('');
  const [hasSaved, setHasSaved] = useState(hasSavedExperiment);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const editRef = useRef<Selection | null>(null);
  const data = useRef(dataset('xor'));
  const frame = timeline.present;
  const event = frame.plan[frame.cursor];
  const lockedRate = frame.cursor > 0 && frame.cursor < frame.plan.length - 1;

  function reset(nextSpecs = specs, nextProblem = problem, nextSeed = seed, network?: Network) {
    setPlaying(false);
    setSelection(null);
    data.current = dataset(nextProblem);
    setTimeline({ past: [], present: createFrame(network ?? createNetwork(nextSpecs, nextSeed), data.current), future: [] });
    setSpecs(nextSpecs);
    setProblem(nextProblem);
    setSeed(nextSeed);
  }
  const advance = useCallback(() => {
    setSelection(null);
    setTimeline(t => {
      try {
        const next = mode === 'learn' ? step(t.present, data.current, rate) : trainEpoch(t.present, data.current, rate);
        return { past: [...t.past, t.present].slice(-500), present: next, future: [] };
      } catch (error) { return { ...t, error: error instanceof Error ? error.message : 'Training paused. Reset the network to continue.' }; }
    });
  }, [mode, rate]);
  useEffect(() => {
    if (!playing || timeline.error) return;
    const interval = window.setInterval(advance, mode === 'learn' ? 950 / speed : 80 / speed);
    return () => window.clearInterval(interval);
  }, [advance, playing, mode, speed, timeline.error]);
  useEffect(() => {
    if (!toast) return;
    const timeout = setTimeout(() => setToast(''), 4000);
    return () => clearTimeout(timeout);
  }, [toast]);
  useEffect(() => {
    if (modal) { setPlaying(false); dialogRef.current?.showModal(); }
    else dialogRef.current?.close();
  }, [modal]);
  function back() {
    setPlaying(false); setSelection(null);
    setTimeline(t => t.past.length ? { past: t.past.slice(0, -1), present: t.past.at(-1)!, future: [t.present, ...t.future] } : t);
  }
  function next() {
    setPlaying(false); setSelection(null);
    if (timeline.future.length) setTimeline(t => ({ past: [...t.past, t.present].slice(-500), present: t.future[0], future: t.future.slice(1) }));
    else advance();
  }
  function changeMode(nextMode: Mode) { setMode(nextMode); setPlaying(false); setSelection(null); setTimeline(t => ({ ...t, future: [] })); }
  function updateLayer(index: number, changes: Partial<LayerSpec>) {
    const updated = specs.map((s, i) => i === index ? { ...s, ...changes } : s);
    if (index === specs.length - 1 && changes.activation === 'softmax') updated[index].size = 2;
    reset(updated);
  }
  function chooseSample(index: number) {
    setPlaying(false); setSelection(null);
    setTimeline(t => ({ past: [...t.past, t.present].slice(-500), present: createFrame(t.present.cursor === t.present.plan.length - 1 ? t.present.network : t.present.trace.source, data.current, index, t.present.samplesSeen, t.present.history), future: [] }));
    setToast('Inspecting this example. Any in-progress sample updates were rewound.');
  }
  function save() {
    try {
      localStorage.setItem('neuron-lab-experiment', JSON.stringify({ version: 1, specs, problem, seed, rate, network: frame.network }));
      setHasSaved(true); setToast('Network, settings, and weights saved in this browser.');
    } catch { setToast('Browser storage is unavailable. Your current experiment is still open.'); }
  }
  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem('neuron-lab-experiment') ?? 'null');
      if (!saved || saved.version !== 1 || !PROBLEMS[saved.problem as Problem] || !Array.isArray(saved.specs) || !Array.isArray(saved.network?.layers)) throw new Error();
      reset(saved.specs, saved.problem, saved.seed, saved.network); setRate(saved.rate); setToast('Saved weights restored. A fresh training history is ready.');
    } catch { setToast('Could not open the saved experiment. Start a new one or save again.'); }
  }
  const activePhaseIndex = PHASES.findIndex(p => p.id === event.phase);
  const parameterTotal = parameterCount(frame.network);
  return <>
    <header className="app-header"><a className="brand" href="./" aria-label="Neuron Lab home"><span className="brand-mark"><NetworkIcon size={23}/></span><span>neuron<span className="brand-light">lab</span><span className="brand-period">.</span></span></a><div className="header-center"><span className="header-divider"/> A little curiosity. A lot of connections.</div><div className="header-actions"><span className="local-badge"><span className="status-dot"/> Runs in your browser</span><button className="quiet-button" onClick={() => setModal('guide')}><BookOpen size={16}/> Quick guide</button><button className="outline-button" onClick={save}><Save size={15}/> Save lab</button>{hasSaved && <button className="quiet-button restore-button" onClick={restore} aria-label="Restore saved"><RotateCcw size={14}/><span>Restore saved</span></button>}</div></header>
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-title"><SlidersIcon/><span>Your experiment</span><span className="version-tag">01</span></div>
        <section className="config-section"><label className="section-label">01 <span>CHOOSE A NETWORK</span></label><div className="model-grid"><button className="model-button active" aria-pressed="true"><NetworkIcon size={20}/><span>ANN</span><small>Feedforward</small></button>{['CNN', 'RNN', 'LSTM'].map((name, i) => <button className="model-button upcoming" key={name} onClick={() => setModal('roadmap')} aria-label={`${name}, planned architecture`}><span className="soon-dot"/>{i === 0 ? <Layers3 size={20}/> : i === 1 ? <RotateCcw size={20}/> : <Cpu size={20}/>}<span>{name}</span><small>Coming later</small></button>)}</div></section>
        <section className="config-section"><label className="section-label" htmlFor="problem">02 <span>PICK A PROBLEM</span></label><div className="select-wrap"><FlaskConical size={16}/><select id="problem" value={problem} onChange={e => reset(specs, e.target.value as Problem)}><option value="xor">XOR logic gate</option><option value="circle">2D points · circle</option><option value="diagonal">2D points · diagonal</option></select><ChevronDown size={14}/></div><p className="field-help">{problem === 'xor' ? 'A classic problem with a nonlinear twist.' : '80 reproducible, labeled training points.'}</p></section>
        <section className="config-section architecture-section"><div className="section-label">03 <span>SHAPE YOUR NETWORK</span></div><div className="architecture-heading"><span>Architecture</span><span className="mono">2 → {specs.map(s => s.size).join(' → ')}</span></div>
          <div className="layer-row fixed-layer"><span className="layer-dot input-dot"/><div><strong>Input layer</strong><small>Set by the problem</small></div><span className="neuron-count">2</span></div>
          <div className="layers-stack">{specs.slice(0, -1).map((spec, i) => <div className="layer-editor" key={i}><div className="layer-row"><span className="layer-dot"/><strong>Hidden {i + 1}</strong><div className="stepper"><button aria-label={`Remove neuron from hidden layer ${i + 1}`} disabled={spec.size <= 1} onClick={() => updateLayer(i, { size: spec.size - 1 })}><Minus size={12}/></button><span>{spec.size}</span><button aria-label={`Add neuron to hidden layer ${i + 1}`} disabled={spec.size >= 12} onClick={() => updateLayer(i, { size: spec.size + 1 })}><Plus size={12}/></button></div><button className="icon-button delete-layer" aria-label={`Remove hidden layer ${i + 1}`} onClick={() => reset(specs.filter((_, k) => k !== i))}><Trash2 size={13}/></button></div><select className="activation-select" aria-label={`Hidden layer ${i + 1} activation`} value={spec.activation} onChange={e => updateLayer(i, { activation: e.target.value as Activation })}>{ACTIVATIONS.map(a => <option key={a}>{a}</option>)}</select></div>)}</div>
          <button className="add-layer" disabled={specs.length >= 7} onClick={() => reset([...specs.slice(0, -1), { size: 3, activation: 'tanh' }, specs.at(-1)!])}><Plus size={14}/> Add hidden layer</button>
          <div className="output-editor"><div className="layer-row"><span className="layer-dot output-dot"/><strong>Output layer</strong><select aria-label="Output neurons" value={specs.at(-1)!.size} onChange={e => updateLayer(specs.length - 1, { size: Number(e.target.value) })}><option value="1" disabled={specs.at(-1)!.activation === 'softmax'}>1 neuron</option><option value="2">2 neurons</option></select></div><select className="activation-select" aria-label="Output activation" value={specs.at(-1)!.activation} onChange={e => updateLayer(specs.length - 1, { activation: e.target.value as Activation })}>{ACTIVATIONS.map(a => <option key={a}>{a}</option>)}</select></div>
          <p className="field-help">Architecture changes reset the weights. Up to 6 hidden layers, 12 neurons each.</p><button className="text-button" onClick={() => { reset(DEFAULT_SPECS, problem, 42); setRate(0.3); }}><RotateCcw size={12}/> Use recommended network</button>
        </section>
        <section className="config-section learning-settings"><label className="section-label" htmlFor="learning-rate">04 <span>TUNE THE LEARNING</span></label><div className="field-line"><label htmlFor="learning-rate">Learning rate <span className="math-inline">η</span></label><output>{rate.toFixed(2)}</output></div><input id="learning-rate" type="range" min="0.01" max="1" step="0.01" value={rate} disabled={lockedRate || playing} onChange={e => { setRate(Number(e.target.value)); setTimeline(t => ({ ...t, future: [] })); }}/><div className="range-labels"><span>Careful</span><span>Adventurous</span></div><p className="field-help">{lockedRate ? 'Finish this sample or reset to change its learning rate.' : 'How big a step to take with each update.'}</p><div className="field-line seed-field"><label htmlFor="seed">Random seed</label><input id="seed" aria-label="Random seed" type="number" min="0" max="99999" value={seed} onChange={e => { const value = Number(e.target.value); if (Number.isInteger(value) && value >= 0 && value <= 99999) reset(specs, problem, value); }}/></div></section>
        <div className="sidebar-note"><span className="note-icon"><Sparkles size={16}/></span><p>No black boxes here.<br/><strong>Every number has a story.</strong></p></div>
      </aside>
      <main className="main-content">
        <div className="page-intro"><div><div className="eyebrow intro-eyebrow"><span/> THE NEURAL NETWORK PLAYGROUND</div><h1>See learning happen<span>.</span></h1><p>Build a network. Follow a signal. Make the math click.</p></div><span className="experiment-badge"><span className="tiny-dot purple"/> Experiment 01 <span>/</span> ANN</span></div>
        <div className="experiment-topbar"><div className="problem-title"><span className="problem-icon"><GitBranch size={18}/></span><div><h2>{PROBLEMS[problem].title}</h2><p>{PROBLEMS[problem].description}</p></div></div><div className="mode-switch" aria-label="Learning mode"><button className={mode === 'learn' ? 'active' : ''} aria-pressed={mode === 'learn'} onClick={() => changeMode('learn')}><BookOpen size={14}/> Learn</button><button className={mode === 'train' ? 'active' : ''} aria-pressed={mode === 'train'} onClick={() => changeMode('train')}><Zap size={14}/> Train</button></div></div>
        <div className="workbench">
          <section className="network-card card"><div className="network-heading"><div><h2>Your neural network <span className="small-chip">Live</span></h2><p>{specs.length + 1} layers <span>·</span> {2 + specs.reduce((s, l) => s + l.size, 0)} neurons <span>·</span> {parameterTotal} parameters</p></div><button className="icon-button" onClick={() => reset()} title="Reset network" aria-label="Reset network"><RotateCcw size={16}/></button></div>
            <div className="phase-track">{PHASES.map((p, i) => <div className={`${i === activePhaseIndex ? 'current' : ''} ${i < activePhaseIndex ? 'complete' : ''}`} key={p.id}><span>{i < activePhaseIndex ? <Check size={11}/> : i + 1}</span><strong>{p.name}</strong>{i < 4 && <span className="phase-connector"/>}</div>)}</div>
            <NetworkView frame={frame} selection={selection} onSelect={s => { setSelection(s); setPlaying(false); }} training={mode === 'train'}/>
            <div className="step-explanation"><span className={`step-icon ${event.phase}`}><PHASE_ICON phase={event.phase}/></span><div><strong>{mode === 'train' ? 'The big picture, one epoch at a time.' : event.phase === 'input' ? 'Start with an example.' : event.phase === 'forward' ? `Forward pass · layer ${event.layer! + 1}, neuron ${event.neuron! + 1}` : event.phase === 'loss' ? 'Compare the prediction with the target.' : event.phase === 'backward' ? `Backpropagation · layer ${event.layer! + 1}, neuron ${event.neuron! + 1}` : `Update ${event.bias ? 'bias' : `weight ${event.input! + 1}`} · layer ${event.layer! + 1}, neuron ${event.neuron! + 1}`}</strong><p>{mode === 'train' ? `Each tick trains on all ${data.current.length} examples. Pause and switch to Learn to look closer.` : event.phase === 'input' ? `Feed x = [${data.current[frame.sampleIndex].x.map(v => format(v, 2)).join(', ')}] into the network. Our target is ${data.current[frame.sampleIndex].y}.` : event.phase === 'forward' ? 'Multiply inputs by weights, add the bias, and apply the activation.' : event.phase === 'loss' ? `This example's loss is ${format(frame.trace.loss, 6)}. Now we find out how to improve it.` : event.phase === 'backward' ? 'Use the chain rule to find how this neuron contributes to the loss.' : 'Only this highlighted parameter changes. All gradients were computed before updates began.'}</p></div></div>
            <div className="playback-bar"><div className="playback-buttons"><button className="control-button" disabled={!timeline.past.length} onClick={back} aria-label="Step backward" title="Step backward"><SkipBack size={17}/></button><button className="play-button" onClick={() => { setSelection(null); setPlaying(v => !v); }} disabled={Boolean(timeline.error)} aria-label={playing && !timeline.error ? 'Pause' : mode === 'learn' ? 'Play learning steps' : 'Start training'}>{playing && !timeline.error ? <Pause size={16} fill="currentColor"/> : <Play size={16} fill="currentColor"/>}<span>{playing && !timeline.error ? 'Pause' : mode === 'learn' ? 'Play' : 'Train'}</span></button><button className="control-button" disabled={Boolean(timeline.error)} onClick={next} aria-label="Step forward" title={mode === 'learn' ? 'Next calculation' : 'Train one epoch'}><SkipForward size={17}/></button><span className="playback-divider"/><label className="speed-control" title="Playback speed"><span className="sr-only">Playback speed</span><select aria-label="Playback speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label></div><span className="step-counter">{mode === 'learn' ? <>Step <strong data-testid="step-number">{frame.cursor + 1}</strong> / {frame.plan.length}</> : <>Epoch <strong>{format(frame.metric.epoch, 1)}</strong></>}</span></div>
          </section>
          <Inspector frame={frame} selection={selection} rate={frame.learningRate ?? rate} onEdit={s => { editRef.current = s; setModal('edit'); }}/>
        </div>
        {timeline.error && <div className="error-banner" role="alert">{timeline.error}<button onClick={() => reset()}>Reset network</button></div>}
        <div className="results-grid"><section className="card data-card"><div className="panel-heading"><h2>{problem === 'xor' ? 'The training data' : 'Decision boundary'}</h2><span className="small-chip">{data.current.length} examples</span></div><DataView frame={frame} data={data.current} problem={problem} onSample={chooseSample}/><div className="data-footer"><MousePointer2 size={12}/>{problem === 'xor' ? 'Click an input to walk through its prediction.' : 'Training points · not a held-out test set'}</div></section><section className="card loss-card"><div className="panel-heading"><h2>A little better, every epoch</h2><span className="legend"><span className="tiny-dot purple"/> Training loss</span></div><div className="metrics-row"><div><span>Mean loss</span><strong data-testid="loss">{format(frame.metric.loss)}</strong></div><div><span>Training accuracy</span><strong className="accuracy" data-testid="accuracy">{Math.round(frame.metric.accuracy * 100)}<small>%</small></strong></div><div><span>Epoch</span><strong data-testid="epoch">{format(frame.metric.epoch, frame.metric.epoch % 1 ? 2 : 0)}</strong></div></div><LossChart frame={frame}/></section></div>
        <footer className="main-footer"><span><span className="status-dot"/> Real math. Real learning. All on your device.</span><button className="text-button" onClick={() => setModal('guide')}>A few things worth knowing <ArrowRight size={13}/></button></footer>
      </main>
    </div>
    {toast && <div className="toast" role="status"><Check size={16}/>{toast}<button aria-label="Dismiss notification" onClick={() => setToast('')}><X size={14}/></button></div>}
    <dialog ref={dialogRef} className="modal" aria-label={modal === 'guide' ? 'Quick guide' : modal === 'roadmap' ? 'Architecture roadmap' : 'Edit neuron parameters'} onCancel={() => setModal(null)} onClick={e => { if (e.target === e.currentTarget) setModal(null); }}><div className="modal-inner"><button className="modal-close icon-button" aria-label="Close dialog" onClick={() => setModal(null)}><X size={20}/></button>
      {modal === 'guide' && <><span className="modal-symbol"><BookOpen size={24}/></span><span className="eyebrow">A FIELD GUIDE FOR CURIOUS MINDS</span><h2>Your first aha moment starts here.</h2><p>A neural network learns by adjusting the strength of its connections. This lab lets you see each adjustment happen.</p><ol className="guide-steps"><li><strong>Start with XOR.</strong> The answer is 1 when the two inputs are different. A hidden layer helps the network learn this nonlinear rule.</li><li><strong>Follow one signal in Learn mode.</strong> Use the next button to inspect each neuron, the loss, the gradients, then every weight and bias update.</li><li><strong>Zoom out in Train mode.</strong> Each tick trains on the whole dataset (one epoch), using one example at a time. Watch the loss and accuracy.</li><li><strong>Go backward, then experiment.</strong> The back button restores exact prior states, including weights and charts. Up to 500 steps are kept in memory. Editing settings starts a new branch.</li></ol><div className="guide-glossary"><div><b>Weight</b><span>How strongly a connection carries a signal.</span></div><div><b>Bias</b><span>An adjustable offset added to a neuron's weighted sum.</span></div><div><b>Activation</b><span>A function that transforms the weighted sum. Nonlinear functions let a network learn nonlinear patterns.</span></div><div><b>Gradient</b><span>How much a small parameter change would change the loss.</span></div></div><p className="fine-print">The loss is half squared error, averaged over examples for the chart. Two outputs use targets [1, 0] and [0, 1]; their predicted class is the larger output. One output uses a 0.5 threshold. Linear and ReLU outputs are raw scores, not probabilities. Accuracy is measured on training data. XOR and 2D inputs fix the input layer at two features. Softmax outputs require two neurons. Numerical values are rounded for display; computations use full precision.</p><button className="primary-button" onClick={() => setModal(null)}>Let's explore <ArrowRight size={16}/></button></>}
      {modal === 'roadmap' && <><span className="modal-symbol"><Layers3 size={24}/></span><span className="eyebrow">THE NEXT CONNECTIONS</span><h2>One architecture at a time.</h2><p>This first release is a fully working feedforward ANN lab. The other architectures are planned, each with its own learning view.</p><div className="roadmap-list"><div><span className="roadmap-tag ready">ANN</span><div><b>Available now</b><p>Editable layers, XOR and 2D classification, full forward and backward calculations.</p></div></div><div><span className="roadmap-tag">CNN</span><div><b>Small image classification</b><p>Explore convolution filters, feature maps, pooling, and image gradients.</p></div></div><div><span className="roadmap-tag">RNN</span><div><b>Learning from sequences</b><p>Unfold the network across time and trace gradients through its hidden states.</p></div></div><div><span className="roadmap-tag">LSTM</span><div><b>A closer look at memory</b><p>Inspect input, forget, and output gates, plus cell-state updates.</p></div></div></div><button className="primary-button" onClick={() => setModal(null)}>Back to the ANN lab <ArrowRight size={16}/></button></>}
      {modal === 'edit' && editRef.current && <ParameterEditor key={`${editRef.current.layer}-${editRef.current.neuron}`} network={frame.network} selected={editRef.current} onApply={network => { reset(specs, problem, seed, network); setModal(null); setToast('Parameters updated. A new experiment starts from these weights.'); }}/>}
    </div></dialog>
  </>;
}
function SlidersIcon() { return <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M2 4h12M2 12h12"/><circle cx="6" cy="4" r="2" fill="white"/><circle cx="10" cy="12" r="2" fill="white"/></svg>; }
function PHASE_ICON({ phase }: { phase: string }) { const Icon = PHASES.find(p => p.id === phase)?.icon ?? ArrowRight; return <Icon size={17}/>; }
function ParameterEditor({ network, selected, onApply }: { network: Network; selected: Selection; onApply: (n: Network) => void }) {
  const { layer: l, neuron: j } = selected;
  const [values, setValues] = useState(network.layers[l].weights[j].map(String));
  const [bias, setBias] = useState(String(network.layers[l].biases[j]));
  return <form onSubmit={e => { e.preventDefault(); const copy = structuredClone(network); copy.layers[l].weights[j] = values.map(Number); copy.layers[l].biases[j] = Number(bias); onApply(copy); }}><span className="eyebrow">TAKE THE CONTROLS</span><h2>Edit a neuron's parameters.</h2><p>Layer {l + 1}, neuron {j + 1}. Applying changes starts a new training history with these values.</p><div className="parameter-fields">{values.map((value, i) => <label key={i}>Weight from input {i + 1}<input aria-label={`Weight from input ${i + 1}`} type="number" step="any" min="-100" max="100" required value={value} onChange={e => setValues(v => v.map((x, k) => k === i ? e.target.value : x))}/></label>)}<label>Bias<input aria-label="Bias" type="number" min="-100" max="100" step="any" required value={bias} onChange={e => setBias(e.target.value)}/></label></div><button className="primary-button" type="submit">Apply parameters <Check size={16}/></button></form>;
}
