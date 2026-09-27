import { useMemo, useRef, useState } from 'react';
import { FlaskConical, MousePointer2 } from 'lucide-react';
import { DEFAULT_WORD, cleanWord, createSequenceModel, generate, seqForward, seqLab, sequenceTask, type SeqFrame, type SeqKind, type SeqModel, type SeqProblem, type SeqTask } from './sequence';
import { SeqView, defaultSeqSelection, type SeqSelection } from './SeqView';
import { SeqInspector } from './SeqInspector';
import { argmax } from './lab';
import { format } from './engine';
import { useI18n } from './i18n';
import { useTimeline } from './useTimeline';
import { LabLayout, LearningSettings, NetworkHeading, PhaseTrack, PlaybackBar, ProblemHeader, ResultsCard, Section, SelectField, StepGuide, Stepper, useFrameSelection, useSaved, type LabProps } from './ui';

const PROBLEMS: SeqProblem[] = ['word', 'memory'];
const DEFAULTS: Record<SeqKind, { rate: number; hidden: number; seed: number }> = { rnn: { rate: 0.3, hidden: 4, seed: 3 }, lstm: { rate: 0.5, hidden: 4, seed: 3 } };
interface Settings { problem: SeqProblem; word: string; length: number; hidden: number; seed: number }

export function SeqLab({ kind, active, picker, notify }: LabProps & { kind: SeqKind }) {
  const { t } = useI18n();
  const T = t.seq;
  const defaults = DEFAULTS[kind];
  const [settings, setSettings] = useState<Settings>({ problem: 'word', word: DEFAULT_WORD, length: 4, hidden: defaults.hidden, seed: defaults.seed });
  const [rate, setRate] = useState(defaults.rate);
  const [draft, setDraft] = useState(DEFAULT_WORD);
  const task = useRef<SeqTask>(sequenceTask('word', DEFAULT_WORD));
  const controls = useTimeline(() => seqLab.createFrame(createSequenceModel(kind, defaults.hidden, task.current.symbols.length, defaults.seed), task.current.data), { step: (f, r) => seqLab.step(f, task.current.data, r), epoch: (f, r) => seqLab.trainEpoch(f, task.current.data, r) }, rate, active);
  const { frame, mode } = controls;
  const [picked, setPicked] = useFrameSelection<SeqSelection>(frame);
  const selection = picked ?? defaultSeqSelection(frame);
  const saved = useSaved(`neuron-lab-${kind}`);
  const symbols = task.current.symbols, data = task.current.data;
  const step = frame.plan[frame.cursor];

  function reset(next: Partial<Settings> = {}, model?: SeqModel) {
    const s = { ...settings, ...next };
    task.current = sequenceTask(s.problem, s.word, s.length);
    controls.reset(seqLab.createFrame(model ?? createSequenceModel(kind, s.hidden, task.current.symbols.length, s.seed), task.current.data));
    setSettings(s);
  }
  function chooseSample(index: number) {
    controls.push(f => seqLab.createFrame(f.cursor === f.plan.length - 1 ? f.model : f.source, data, index, f.samplesSeen, f.history));
    notify(t.common.sampleToast);
  }
  function save() {
    try { saved.save({ version: 1, kind, settings, rate, model: frame.model }); notify(t.common.savedToast); }
    catch { notify(t.common.storageUnavailable); }
  }
  function restore() {
    try {
      const s = saved.load() as { version: number; kind: SeqKind; settings: Settings; rate: number; model: SeqModel } | null;
      if (!s || s.version !== 1 || s.kind !== kind || !PROBLEMS.includes(s.settings?.problem) || typeof s.model?.params !== 'object') throw new Error();
      reset(s.settings, s.model); setDraft(s.settings.word); setRate(s.rate); notify(t.common.restoredToast);
    } catch { notify(t.common.restoreFailed); }
  }
  const S = T.step;
  const sequenceText = settings.problem === 'word' ? settings.word : data[frame.sampleIndex].x.join('');
  const targets = data[frame.sampleIndex].y.filter(y => y !== null).length;
  const [title, text] = mode === 'train' ? [S.trainTitle, S.trainText(data.length)]
    : step.phase === 'input' ? [S.inputTitle, settings.problem === 'word' ? S.inputWord(sequenceText) : S.inputMemory(sequenceText)]
    : step.phase === 'forward' ? [S.forwardTitle(step.t! + 1), kind === 'lstm' ? S.forwardLstm : S.forwardRnn]
    : step.phase === 'loss' ? [S.lossTitle, S.lossText(targets, format(frame.trace.loss, 5))]
    : step.phase === 'backward' ? [S.backwardTitle(step.t! + 1), S.backwardText]
    : [S.updateTitle(T.groups[step.group as keyof typeof T.groups]), S.updateText];
  const wordError = draft !== settings.word && !cleanWord(draft);
  const sidebar = <>
    <Section number={2} label={t.sections.problem} htmlFor={`${kind}-problem`} tour="problem">
      <SelectField id={`${kind}-problem`} value={settings.problem} onChange={v => reset({ problem: v as SeqProblem })} icon={<FlaskConical size={17}/>}>{PROBLEMS.map(p => <option key={p} value={p}>{T.problems[p].option}</option>)}</SelectField>
      <p className="field-help">{T.problems[settings.problem].help}</p>
      {settings.problem === 'word' ? <form className="word-form" onSubmit={e => { e.preventDefault(); const w = cleanWord(draft); if (w) { setDraft(w); reset({ word: w }); } }}>
        <label htmlFor={`${kind}-word`}>{T.word}</label>
        <div className="inline-field"><input id={`${kind}-word`} value={draft} maxLength={12} onChange={e => setDraft(e.target.value)} aria-invalid={wordError} autoComplete="off" spellCheck={false}/><button type="submit" className="ghost-button" disabled={!cleanWord(draft) || cleanWord(draft) === settings.word}>{T.apply}</button></div>
        {wordError && <p className="field-error">{T.wordInvalid}</p>}
      </form> : <div className="field-block">
        <div className="field-line"><label htmlFor={`${kind}-length`}>{T.length}</label><output>{settings.length}</output></div>
        <input id={`${kind}-length`} type="range" min="2" max="8" step="1" value={settings.length} onChange={e => reset({ length: Number(e.target.value) })}/>
      </div>}
    </Section>
    <Section number={3} label={t.sections.shape}>
      <div className="architecture-heading"><span className="mono pipeline-text">{symbols.length} → {kind.toUpperCase()}({settings.hidden}) → {symbols.length}</span></div>
      <div className="layer-editor"><div className="layer-row"><span className="layer-dot"/><strong>{T.hidden}</strong><Stepper value={settings.hidden} min={1} max={8} onChange={hidden => reset({ hidden })} decrease={T.removeHidden} increase={T.addHidden}/></div><p className="field-help">{T.hiddenHelp}</p></div>
      <p className="field-help">{t.common.architectureResets}</p>
    </Section>
    <LearningSettings number={4} rate={rate} onRate={v => { setRate(v); controls.clearFuture(); }} locked={controls.lockedRate || controls.playing} seed={settings.seed} onSeed={v => reset({ seed: v })} onRecommended={() => { reset({ hidden: defaults.hidden, seed: defaults.seed }); setRate(defaults.rate); notify(t.common.recommendedToast); }}/>
  </>;
  return <LabLayout id={kind} active={active} picker={picker} sidebar={sidebar}>
    <ProblemHeader title={T.problems[settings.problem].title} description={T.problems[settings.problem].description} mode={mode} onMode={controls.changeMode}/>
    <div className="workbench">
      <section className="network-card card">
        <NetworkHeading title={T.networkTitle[kind]} stats={[`${settings.hidden} ${T.hidden.toLowerCase()}`, `${data[frame.sampleIndex].x.length} × t`, t.common.parameters(seqLab.parameterCount(frame.model))]} onReset={() => reset()} onSave={save} onRestore={restore} hasSaved={saved.hasSaved}/>
        <PhaseTrack phase={step.phase} mode={mode}/>
        <SeqView frame={frame} symbols={symbols} selection={selection} onSelect={s => { setPicked(s); controls.setPlaying(false); }} training={mode === 'train'}/>
        <StepGuide phase={step.phase} title={title} text={text} frame={frame} mode={mode}/>
        <PlaybackBar controls={controls}/>
      </section>
      <SeqInspector frame={frame} selection={selection} picked={picked !== null} symbols={symbols} rate={frame.learningRate ?? rate}/>
    </div>
    {controls.timeline.error && <div className="error-banner" role="alert">{t.common.diverged}<button onClick={() => reset()}>{t.common.resetNetwork}</button></div>}
    <div className="results-grid" data-tour="results">
      {settings.problem === 'word' ? <WordData frame={frame} symbols={symbols} word={settings.word}/> : <MemoryData frame={frame} data={data} onSample={chooseSample}/>}
      <ResultsCard frame={frame}/>
    </div>
  </LabLayout>;
}
function WordData({ frame, symbols, word }: { frame: SeqFrame; symbols: string[]; word: string }) {
  const { t } = useI18n();
  const T = t.seq.data;
  const sequence = { x: [...word].slice(0, -1).map(c => symbols.indexOf(c)), y: [...word].slice(1).map(c => symbols.indexOf(c)) };
  const guesses = useMemo(() => seqForward(frame.model, sequence).map(s => argmax(s.probs)), [frame.model, word]); // eslint-disable-line react-hooks/exhaustive-deps
  const written = useMemo(() => generate(frame.model, symbols.indexOf(word[0]), word.length), [frame.model, word, symbols]);
  return <section className="card data-card">
    <div className="panel-heading"><h2>{T.sequenceTitle}</h2><span className="small-chip">“{word}”</span></div>
    <div className="table-scroll"><table className="data-table word-table"><tbody>
      <tr><th>x</th>{sequence.x.map((c, i) => <td key={i}><span className="token">{symbols[c]}</span></td>)}</tr>
      <tr><th>{T.next}</th>{sequence.y.map((c, i) => <td key={i}><span className="token target">{symbols[c]}</span></td>)}</tr>
      <tr><th>{T.guess}</th>{guesses.map((c, i) => <td key={i}><span className={`token ${c === sequence.y[i] ? 'correct' : 'incorrect'}`}>{symbols[c]}</span></td>)}</tr>
    </tbody></table></div>
    <div className="write-box"><strong>{T.writeTitle}</strong><p>{T.writeText(word[0])}</p><div className="written" data-testid="generated">{written.map((c, i) => <span key={i} className={`token ${symbols[c] === word[i] ? 'correct' : 'incorrect'}`}>{symbols[c]}</span>)}</div></div>
    <div className="data-footer"><MousePointer2 size={14}/>{T.wordNote}</div>
  </section>;
}
function MemoryData({ frame, data, onSample }: { frame: SeqFrame; data: SeqTask['data']; onSample: (i: number) => void }) {
  const { t } = useI18n();
  const T = t.seq.data;
  const predictions = useMemo(() => data.map(s => seqForward(frame.model, s).at(-1)!.probs), [frame.model, data]);
  return <section className="card data-card">
    <div className="panel-heading"><h2>{T.sequencesTitle}</h2><span className="small-chip">{t.common.examples(data.length)}</span></div>
    <div className="table-scroll tall"><table className="data-table"><thead><tr><th>x</th><th>{T.first}</th><th>p(1)</th><th aria-label={t.common.correct}/></tr></thead><tbody>{data.map((s, i) => {
      const p = predictions[i], correct = argmax(p) === s.x[0];
      return <tr key={i} className={frame.sampleIndex === i ? 'current-sample' : ''} onClick={() => onSample(i)}><td><button className="mono" aria-label={T.inspect(i + 1)} onClick={e => { e.stopPropagation(); onSample(i); }}><b>{s.x[0]}</b>{s.x.slice(1).join('')}</button></td><td><span className={`target-value class-${s.x[0]}`}>{s.x[0]}</span></td><td className="mono">{format(p[1], 3)}</td><td><span className={`prediction-dot ${correct ? 'correct' : 'incorrect'}`} title={correct ? t.common.correct : t.common.incorrect}/></td></tr>;
    })}</tbody></table></div>
    <div className="data-footer"><MousePointer2 size={14}/>{T.memoryNote}</div>
  </section>;
}
