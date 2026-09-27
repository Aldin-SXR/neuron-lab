import { createContext, useContext, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { ArrowDown, ArrowRight, BookOpen, Check, ChevronDown, CircleHelp, Cpu, GitBranch, Image as ImageIcon, Lightbulb, Minus, Network as NetworkIcon, Pause, Play, Plus, Repeat, RotateCcw, Save, SkipBack, SkipForward, X, Zap } from 'lucide-react';
import type { LabFrame, Metric, Phase } from './lab';
import { useI18n } from './i18n';
import type { Mode, PlaybackControls } from './useTimeline';

export type Architecture = 'ann' | 'cnn' | 'rnn' | 'lstm';
export const ARCHITECTURES: Architecture[] = ['ann', 'cnn', 'rnn', 'lstm'];
const ARCH_ICONS = { ann: NetworkIcon, cnn: ImageIcon, rnn: Repeat, lstm: Cpu };
export const PHASES: { id: Phase; icon: typeof ArrowRight }[] = [
  { id: 'input', icon: GitBranch }, { id: 'forward', icon: ArrowRight }, { id: 'loss', icon: CircleHelp }, { id: 'backward', icon: ArrowDown }, { id: 'update', icon: Zap },
];
export interface LabProps { active: boolean; picker: ReactNode; notify: (message: string) => void }

/**
 * Arrow-key navigation for a group of mutually exclusive buttons (radios or tabs):
 * choose the neighbour and move focus to it. Handled keys never reach the lesson shortcuts.
 */
export function rovingKeys<T>(options: T[], value: T, choose: (v: T) => void) {
  return (e: KeyboardEvent<HTMLElement>) => {
    const step = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
    const index = e.key === 'Home' ? 0 : e.key === 'End' ? options.length - 1 : step ? (options.indexOf(value) + step + options.length) % options.length : -1;
    if (index < 0) return;
    e.preventDefault();
    choose(options[index]);
    const buttons = e.currentTarget.querySelectorAll<HTMLElement>('[role=radio], [role=tab]');
    buttons[index]?.focus();
  };
}
/** Set when the picker itself switched labs, so the newly shown lab's picker takes over focus. */
let pickerHadFocus = false;
export function ArchitecturePicker({ value, onChange }: { value: Architecture; onChange: (a: Architecture) => void }) {
  const { t } = useI18n();
  const group = useRef<HTMLDivElement>(null);
  // Each lab renders its own picker; the one that was used is about to be hidden.
  const choose = (a: Architecture) => { if (a !== value) { pickerHadFocus = true; onChange(a); } };
  useEffect(() => {
    if (!pickerHadFocus || group.current?.closest<HTMLElement>('.lab')?.hidden !== false) return;
    pickerHadFocus = false;
    group.current.querySelector<HTMLElement>('[aria-checked=true]')?.focus();
  }, [value]);
  return <section className="config-section" data-tour="architecture">
    <div className="section-label"><span className="section-number" aria-hidden>1</span>{t.arch.title}</div>
    <div className="model-grid" role="radiogroup" aria-label={t.arch.title} ref={group} onKeyDown={rovingKeys(ARCHITECTURES, value, choose)}>{ARCHITECTURES.map(a => {
      const Icon = ARCH_ICONS[a];
      return <button key={a} role="radio" aria-checked={value === a} tabIndex={value === a ? 0 : -1} className={`model-button ${value === a ? 'active' : ''}`} onClick={() => choose(a)} title={t.arch[a].description}>
        <Icon size={20}/><span className="model-name">{t.arch[a].name}</span><small>{t.arch[a].short}</small>
      </button>;
    })}</div>
  </section>;
}
export function Section({ number, label, htmlFor, tour, className = '', children }: { number: number; label: string; htmlFor?: string; tour?: string; className?: string; children: ReactNode }) {
  const Label = htmlFor ? 'label' : 'div';
  return <section className={`config-section ${className}`} data-tour={tour}><Label className="section-label" htmlFor={htmlFor}><span className="section-number" aria-hidden>{number}</span>{label}</Label>{children}</section>;
}
export const ShellContext = createContext<{ openGuide: () => void; arch: Architecture; textScale: number }>({ openGuide: () => {}, arch: 'ann', textScale: 1 });
export function LabLayout({ active, picker, sidebar, children, id }: { active: boolean; picker: ReactNode; sidebar: ReactNode; children: ReactNode; id: Architecture }) {
  const { t } = useI18n();
  const { openGuide } = useContext(ShellContext);
  return <div className="app-layout lab" hidden={!active} data-lab={id}>
    <aside className="sidebar"><div className="sidebar-title"><span>{t.app.experiment}</span></div>{picker}{sidebar}</aside>
    <main className="main-content">
      <div className="page-intro"><h1>{t.arch[id].long} <span>({t.arch[id].name})</span></h1><p>{t.arch[id].description}</p></div>
      {children}
      <footer className="main-footer"><span className="shortcut-note">{t.app.shortcuts}</span><button className="text-button" onClick={openGuide}>{t.app.footerLink} <ArrowRight size={14}/></button></footer>
    </main>
  </div>;
}
export function SelectField({ id, value, onChange, children, icon, label }: { id?: string; value: string | number; onChange: (value: string) => void; children: ReactNode; icon?: ReactNode; label?: string }) {
  return <div className="select-wrap">{icon}<select id={id} aria-label={label} value={value} onChange={e => onChange(e.target.value)}>{children}</select><ChevronDown size={16}/></div>;
}
export function Stepper({ value, min, max, onChange, decrease, increase }: { value: number; min: number; max: number; onChange: (v: number) => void; decrease: string; increase: string }) {
  return <div className="stepper"><button aria-label={decrease} title={decrease} disabled={value <= min} onClick={() => onChange(value - 1)}><Minus size={14}/></button><span>{value}</span><button aria-label={increase} title={increase} disabled={value >= max} onClick={() => onChange(value + 1)}><Plus size={14}/></button></div>;
}
export function ProblemHeader({ title, description, mode, onMode }: { title: string; description: string; mode: Mode; onMode: (m: Mode) => void }) {
  const { t } = useI18n();
  return <div className="experiment-topbar">
    <div className="problem-title"><span className="problem-icon"><GitBranch size={20}/></span><div><h2>{title}</h2><p>{description}</p></div></div>
    <div className="mode-switch" role="group" aria-label={t.modes.label} data-tour="mode">
      <button className={mode === 'learn' ? 'active' : ''} aria-pressed={mode === 'learn'} onClick={() => onMode('learn')}><BookOpen size={16}/><span><b>{t.modes.learn}</b><small>{t.modes.learnHint}</small></span></button>
      <button className={mode === 'train' ? 'active' : ''} aria-pressed={mode === 'train'} onClick={() => onMode('train')}><Zap size={16}/><span><b>{t.modes.train}</b><small>{t.modes.trainHint}</small></span></button>
    </div>
  </div>;
}
export function PhaseTrack({ phase, mode }: { phase: Phase; mode: Mode }) {
  const { t } = useI18n();
  const active = mode === 'train' ? -1 : PHASES.findIndex(p => p.id === phase);
  return <ol className="phase-track" aria-label={t.phases[phase]}>{PHASES.map((p, i) => <li className={`${i === active ? 'current' : ''} ${i < active ? 'complete' : ''}`} key={p.id} aria-current={i === active ? 'step' : undefined}><span className="phase-dot">{i < active ? <Check size={12}/> : i + 1}</span><strong>{t.phases[p.id]}</strong></li>)}</ol>;
}
export function StepGuide({ phase, title, text, frame, mode }: { phase: Phase; title: string; text: string; frame: LabFrame; mode: Mode }) {
  const { t } = useI18n();
  const Icon = PHASES.find(p => p.id === phase)?.icon ?? ArrowRight;
  const hint = mode === 'train' ? t.hints.train : frame.cursor === 0 ? t.hints.start : frame.cursor === frame.plan.length - 1 ? t.hints.sampleDone : t.hints.continue;
  return <div className="step-explanation" aria-live="polite">
    <span className={`step-icon ${mode === 'train' ? 'update' : phase}`}><Icon size={18}/></span>
    <div><strong>{title}</strong><p>{text}</p><p className="step-hint"><Lightbulb size={14}/> {hint}</p></div>
  </div>;
}
export function PlaybackBar({ controls }: { controls: PlaybackControls }) {
  const { t, num } = useI18n();
  const { mode, playing, timeline, frame } = controls;
  const running = playing && !timeline.error;
  return <div className="playback-bar" data-tour="playback">
    <div className="playback-buttons">
      <button className="control-button" disabled={!controls.canGoBack} onClick={controls.back} aria-label={t.playback.backAria} title={`${t.playback.back} (←)`}><SkipBack size={17}/><span>{t.playback.back}</span></button>
      <button className="next-button" disabled={Boolean(timeline.error)} onClick={controls.next} aria-label={t.playback.nextAria} title={`${mode === 'learn' ? t.playback.next : t.playback.nextEpoch} (→)`}><span>{mode === 'learn' ? t.playback.next : t.playback.nextEpoch}</span><SkipForward size={17}/></button>
      <button className="play-button" onClick={controls.togglePlay} disabled={Boolean(timeline.error)} aria-label={running ? t.playback.pause : mode === 'learn' ? t.playback.playAria : t.playback.trainAria} title={`${running ? t.playback.pause : mode === 'learn' ? t.playback.play : t.playback.train} (space)`}>{running ? <Pause size={16} fill="currentColor"/> : <Play size={16} fill="currentColor"/>}<span>{running ? t.playback.pause : mode === 'learn' ? t.playback.play : t.playback.train}</span></button>
      <label className="speed-control" title={t.playback.speed}><span className="sr-only">{t.playback.speed}</span><select aria-label={t.playback.speed} value={controls.speed} onChange={e => controls.setSpeed(Number(e.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
    </div>
    <span className="step-counter">{mode === 'learn' ? <>{t.playback.step} <strong data-testid="step-number">{frame.cursor + 1}</strong> {t.playback.of} {frame.plan.length}</> : <>{t.playback.epoch} <strong>{num(frame.metric.epoch, 1)}</strong></>}</span>
  </div>;
}
export function ErrorBanner({ controls, onReset }: { controls: PlaybackControls; onReset: () => void }) {
  const { t } = useI18n();
  const error = controls.timeline.error;
  if (!error) return null;
  return <div className="error-banner" role="alert"><span>{error === 'diverged' ? t.common.diverged : t.common.unexpected}</span><span className="banner-actions">
    {controls.canGoBack && <button onClick={controls.back}><SkipBack size={15}/> {t.playback.back}</button>}
    <button onClick={onReset}>{t.common.resetNetwork}</button>
  </span></div>;
}
export function NetworkHeading({ title, stats, onReset, onSave, onRestore, hasSaved }: { title: string; stats: string[]; onReset: () => void; onSave: () => void; onRestore: () => void; hasSaved: boolean }) {
  const { t } = useI18n();
  return <div className="network-heading">
    <div><h2>{title}</h2><p>{stats.map((s, i) => <span key={i}>{i > 0 && <span className="dot-sep">·</span>}{s}</span>)}</p></div>
    <div className="heading-actions">
      <button className="ghost-button" onClick={onSave} aria-label={t.common.saveAria} title={t.common.saveAria}><Save size={16}/><span>{t.common.save}</span></button>
      {hasSaved && <button className="ghost-button" onClick={onRestore} aria-label={t.common.restoreAria} title={t.common.restoreAria}><RotateCcw size={16}/><span>{t.common.restore}</span></button>}
      <button className="icon-button" onClick={onReset} title={t.common.resetHelp} aria-label={t.common.resetNetwork}><RotateCcw size={17}/></button>
    </div>
  </div>;
}
export function LearningSettings({ number, rate, onRate, locked, seed, onSeed, onRecommended }: { number: number; rate: number; onRate: (v: number) => void; locked: boolean; seed: number; onSeed: (v: number) => void; onRecommended: () => void }) {
  const { t, num } = useI18n();
  const id = useId();
  return <Section number={number} label={t.sections.tune} className="learning-settings" tour="settings">
    <div className="field-line"><label htmlFor={`${id}-rate`}>{t.common.learningRate} <span className="math-inline">η</span></label><output>{num(rate, 2)}</output></div>
    <input id={`${id}-rate`} aria-label={t.common.learningRate} type="range" min="0.01" max="1" step="0.01" value={rate} disabled={locked} onChange={e => onRate(Number(e.target.value))}/>
    <div className="range-labels"><span>{t.common.careful}</span><span>{t.common.adventurous}</span></div>
    <p className="field-help">{locked ? t.common.rateLocked : t.common.rateHelp}</p>
    <details className="advanced"><summary>{t.sections.advanced}</summary>
      <div className="field-line seed-field"><label htmlFor={`${id}-seed`}>{t.common.seed}</label><input id={`${id}-seed`} aria-label={t.common.seed} type="number" min="0" max="99999" value={seed} onChange={e => { const v = Number(e.target.value); if (Number.isInteger(v) && v >= 0 && v <= 99999) onSeed(v); }}/></div>
      <p className="field-help">{t.common.seedHelp}</p>
    </details>
    <button className="text-button recommended" onClick={onRecommended}><RotateCcw size={14}/> {t.common.recommended}</button>
  </Section>;
}
export function ResultsCard({ frame, accuracyLabel }: { frame: LabFrame; accuracyLabel?: string }) {
  const { t, num } = useI18n();
  return <section className="card loss-card" data-tour="results-loss">
    <div className="panel-heading"><h2>{t.common.lossTitle}</h2><span className="legend"><span className="tiny-dot purple"/> {t.common.trainingLoss}</span></div>
    <div className="metrics-row">
      <div><span>{t.common.meanLoss}</span><strong data-testid="loss">{num(frame.metric.loss)}</strong></div>
      <div><span>{accuracyLabel ?? t.common.accuracy}</span><strong className="accuracy" data-testid="accuracy">{Math.round(frame.metric.accuracy * 100)}<small>%</small></strong></div>
      <div><span>{t.common.epoch}</span><strong data-testid="epoch">{num(frame.metric.epoch, frame.metric.epoch % 1 ? 2 : 0)}</strong></div>
    </div>
    <LossChart history={frame.history} metric={frame.metric}/>
    <p className="card-note">{t.common.lossHelp}</p>
  </section>;
}
export function LossChart({ history, metric }: { history: Metric[]; metric: Metric }) {
  const { t, num } = useI18n();
  const gradient = `loss-fill-${useId().replace(/:/g, '')}`;
  const max = Math.max(0.01, ...history.map(m => m.loss)) * 1.15;
  const points = history.map((m, i) => `${48 + i * 452 / Math.max(history.length - 1, 1)},${125 - m.loss / max * 102}`).join(' ');
  return <div className="loss-plot"><svg viewBox="0 0 530 163" role="img" aria-label={t.common.lossAria(num(metric.loss), num(metric.epoch, 2))}>
    {[0, 0.5, 1].map(r => <g key={r}><line x1="48" y1={125 - r * 102} x2="505" y2={125 - r * 102} className="grid-line"/><text x="40" y={129 - r * 102} className="chart-label" textAnchor="end">{num(max * r, 2)}</text></g>)}
    <defs> <linearGradient id={gradient} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#7c66d2" stopOpacity="0.22"/><stop offset="100%" stopColor="#7c66d2" stopOpacity="0"/></linearGradient></defs>
    {history.length > 1 && <polygon points={`48,125 ${points} 500,125`} fill={`url(#${gradient})`}/>}
    <polyline points={points} fill="none" stroke="#7057c8" strokeWidth="2.5" strokeLinejoin="round"/>
    {history.length === 1 && <circle cx="48" cy={125 - history[0].loss / max * 102} r="4" fill="#7057c8"/>}
    <text x="48" y="152" className="chart-label">{num(history[0].epoch, 0)}</text><text x="500" y="152" className="chart-label" textAnchor="end">{num(history.at(-1)!.epoch, 1)}</text><text x="274" y="152" className="chart-label" textAnchor="middle">{t.common.epochs}</text>
  </svg>{history.length === 1 && <span className="chart-empty">{t.common.lossEmpty}</span>}</div>;
}
export function Formula({ caption, math, children, result, neutral }: { caption: string; math?: ReactNode; children?: ReactNode; result?: ReactNode; neutral?: boolean }) {
  return <div className={`formula-box ${neutral ? 'neutral' : ''}`}><div className="formula-caption">{caption}</div>{math && <div className="math">{math}</div>}{children && <div className="numeric-formula">{children}</div>}{result !== undefined && <div className="formula-result">{result}</div>}</div>;
}
export function Tip({ children }: { children: ReactNode }) { return <div className="tip"><Lightbulb size={16}/><p>{children}</p></div>; }
export function FlowArrow() { return <ArrowDown className="flow-arrow" size={18}/>; }
export function InspectorShell({ children }: { children: ReactNode }) {
  const { t } = useI18n();
  return <aside className="inspector card" data-tour="inspector"><div className="panel-heading"><Lightbulb size={18}/><h2>{t.common.inside}</h2></div><div className="inspector-content">{children}</div></aside>;
}
/** Diverging color for a signed value: purple for positive, sand for negative. */
export function valueColor(value: number, scale = 1) {
  const strength = Math.min(1, Math.abs(value) / scale);
  return `color-mix(in srgb, ${value >= 0 ? '#7a62d0' : '#d39a55'} ${Math.round(strength * 85)}%, white)`;
}
export function inkColor(value: number, scale = 1) { return Math.abs(value) / scale > 0.55 ? '#fff' : '#3b3450'; }
/** Keeps storage access safe in private windows and when storage is blocked. */
export function useSaved(key: string) {
  const [hasSaved, setHasSaved] = useState(() => { try { return Boolean(localStorage.getItem(key)); } catch { return false; } });
  return {
    hasSaved,
    save(value: unknown) { localStorage.setItem(key, JSON.stringify(value)); setHasSaved(true); },
    load(): unknown { return JSON.parse(localStorage.getItem(key) ?? 'null'); },
  };
}
/** Selection that automatically clears whenever the displayed frame changes. */
export function useFrameSelection<S>(frame: unknown) {
  const [picked, setPicked] = useState<{ frame: unknown; value: S } | null>(null);
  const selection = picked && picked.frame === frame ? picked.value : null;
  return [selection, (value: S | null) => setPicked(value === null ? null : { frame, value })] as const;
}
export function useZoom() {
  const [zoom, setZoom] = useState(1);
  return { zoom, zoomIn: () => setZoom(z => Math.min(1.8, z + 0.15)), zoomOut: () => setZoom(z => Math.max(0.6, z - 0.15)), reset: () => setZoom(1) };
}
export function ZoomControls({ zoom }: { zoom: ReturnType<typeof useZoom> }) {
  const { t } = useI18n();
  return <div className="zoom-controls"><button title={t.common.zoomOut} aria-label={t.common.zoomOut} onClick={zoom.zoomOut}><Minus size={15}/></button><span>{Math.round(zoom.zoom * 100)}%</span><button title={t.common.zoomIn} aria-label={t.common.zoomIn} onClick={zoom.zoomIn}><Plus size={15}/></button><button title={t.common.zoomReset} aria-label={t.common.zoomReset} onClick={zoom.reset}><RotateCcw size={14}/></button></div>;
}
export function Dialog({ open, onClose, label, children, className = '' }: { open: boolean; onClose: () => void; label: string; children: ReactNode; className?: string }) {
  const { t } = useI18n();
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    else if (!open && dialog.open) dialog.close();
  }, [open]);
  return <dialog ref={ref} className={`modal ${className}`} aria-label={label} onCancel={e => { e.preventDefault(); onClose(); }} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
    {open && <div className="modal-inner"><button className="modal-close icon-button" aria-label={t.app.close} onClick={onClose}><X size={20}/></button>{children}</div>}
  </dialog>;
}
