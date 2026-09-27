import { useCallback, useEffect, useState } from 'react';
import { ArrowRight, BookOpen, Check, Compass, Cpu, Image as ImageIcon, Minus, Network as NetworkIcon, Plus, Repeat, X } from 'lucide-react';
import { AnnLab } from './AnnLab';
import { CnnLab } from './CnnLab';
import { SeqLab } from './SeqLab';
import { Tour } from './Tour';
import { LANGUAGES, useI18n, type Lang } from './i18n';
import { ARCHITECTURES, ArchitecturePicker, Dialog, ShellContext, type Architecture } from './ui';

const TEXT_SIZES = [1, 1.125, 1.25];
const ARCH_ICONS = { ann: NetworkIcon, cnn: ImageIcon, rnn: Repeat, lstm: Cpu };
function stored(key: string) { try { return localStorage.getItem(key); } catch { return null; } }
function store(key: string, value: string) { try { localStorage.setItem(key, value); } catch { /* Preferences simply won't persist. */ } }

export default function App() {
  const { t, lang, setLang } = useI18n();
  const [arch, setArch] = useState<Architecture>(() => { const a = stored('neuron-lab-arch'); return ARCHITECTURES.includes(a as Architecture) ? a as Architecture : 'ann'; });
  const [visited, setVisited] = useState<Set<Architecture>>(() => new Set([arch]));
  const [modal, setModal] = useState<'welcome' | 'guide' | null>(() => stored('neuron-lab-welcomed') ? null : 'welcome');
  const [touring, setTouring] = useState(false);
  // Each notification gets its own id so repeating the same message restarts its timer.
  const [toast, setToast] = useState<{ text: string; id: number } | null>(null);
  const notify = useCallback((text: string) => setToast(previous => ({ text, id: (previous?.id ?? 0) + 1 })), []);
  const [textSize, setTextSize] = useState(() => { const v = Number(stored('neuron-lab-text-size')); return TEXT_SIZES.includes(v) ? v : 1; });
  useEffect(() => {
    document.documentElement.style.fontSize = `${textSize * 100}%`;
    document.documentElement.style.setProperty('--text-scale', String(textSize));
    store('neuron-lab-text-size', String(textSize));
  }, [textSize]);
  useEffect(() => {
    if (!toast) return;
    const timeout = setTimeout(() => setToast(null), 4500);
    return () => clearTimeout(timeout);
  }, [toast]);
  function choose(a: Architecture) { setArch(a); setVisited(v => new Set(v).add(a)); store('neuron-lab-arch', a); }
  function closeWelcome(tour: boolean) { store('neuron-lab-welcomed', '1'); setModal(null); if (tour) setTouring(true); }
  const openGuide = useCallback(() => setModal('guide'), []);
  const picker = <ArchitecturePicker value={arch} onChange={choose}/>;
  const props = (a: Architecture) => ({ active: arch === a, picker, notify });
  const sizeIndex = TEXT_SIZES.indexOf(textSize);
  return <ShellContext.Provider value={{ openGuide, arch, textScale: textSize }}>
    <header className="app-header">
      <a className="brand" href="./" aria-label={t.app.home}><span className="brand-mark"><NetworkIcon size={22}/></span><span>neuron<span className="brand-light">lab</span><span className="brand-period">.</span></span></a>
      <div className="header-actions">
        <div className="segmented lang-switch" role="group" aria-label={t.app.language} data-tour="language">
          {(Object.keys(LANGUAGES) as Lang[]).map(l => <button key={l} aria-pressed={lang === l} className={lang === l ? 'active' : ''} onClick={() => setLang(l)} title={LANGUAGES[l].langName} lang={l}>{LANGUAGES[l].langShort}</button>)}
        </div>
        <div className="segmented text-size" role="group" aria-label={t.app.textSize}>
          <button aria-label={t.app.textSmaller} title={t.app.textSmaller} disabled={sizeIndex <= 0} onClick={() => setTextSize(TEXT_SIZES[sizeIndex - 1])}><Minus size={13}/>A</button>
          <button aria-label={t.app.textLarger} title={t.app.textLarger} disabled={sizeIndex >= TEXT_SIZES.length - 1} onClick={() => setTextSize(TEXT_SIZES[sizeIndex + 1])}><Plus size={13}/>A</button>
        </div>
        <button className="quiet-button" onClick={() => setTouring(true)} aria-label={t.app.tourLong}><Compass size={17}/><span>{t.app.tour}</span></button>
        <button className="quiet-button" onClick={openGuide} aria-label={t.app.guide}><BookOpen size={17}/><span>{t.app.guide}</span></button>
      </div>
    </header>
    {visited.has('ann') && <AnnLab {...props('ann')}/>}
    {visited.has('cnn') && <CnnLab {...props('cnn')}/>}
    {visited.has('rnn') && <SeqLab kind="rnn" {...props('rnn')}/>}
    {visited.has('lstm') && <SeqLab kind="lstm" {...props('lstm')}/>}
    {toast && <div className="toast" role="status"><Check size={18}/>{toast.text}<button aria-label={t.app.dismiss} onClick={() => setToast(null)}><X size={16}/></button></div>}
    {touring && <Tour onClose={() => setTouring(false)}/>}
    <Dialog open={modal === 'welcome'} onClose={() => closeWelcome(false)} label={t.welcome.title} className="welcome-modal">
      <span className="modal-symbol"><NetworkIcon size={26}/></span>
      <span className="eyebrow">{t.welcome.eyebrow}</span>
      <h2>{t.welcome.title}</h2>
      <p>{t.welcome.body}</p>
      <div className="welcome-language"><span>{t.welcome.language}</span><div className="segmented">{(Object.keys(LANGUAGES) as Lang[]).map(l => <button key={l} aria-pressed={lang === l} className={lang === l ? 'active' : ''} onClick={() => setLang(l)} lang={l}>{LANGUAGES[l].langName}</button>)}</div></div>
      <ol className="welcome-points">{t.welcome.points.map(([title, text], i) => <li key={i}><span>{i + 1}</span><div><b>{title}</b><p>{text}</p></div></li>)}</ol>
      <div className="modal-actions"><button className="primary-button" autoFocus onClick={() => closeWelcome(true)}>{t.welcome.tour} <ArrowRight size={17}/></button><button className="ghost-button" onClick={() => closeWelcome(false)}>{t.welcome.skip}</button></div>
    </Dialog>
    <Dialog open={modal === 'guide'} onClose={() => setModal(null)} label={t.app.guide}>
      <span className="modal-symbol"><BookOpen size={26}/></span>
      <span className="eyebrow">{t.guide.eyebrow}</span>
      <h2>{t.guide.title}</h2>
      <p>{t.guide.intro}</p>
      <ol className="guide-steps">{t.guide.steps.map(([b, text], i) => <li key={i}><strong>{b}</strong> {text}</li>)}</ol>
      <h3 className="modal-subtitle">{t.guide.architectures}</h3>
      <div className="roadmap-list">{ARCHITECTURES.map(a => { const Icon = ARCH_ICONS[a]; return <button key={a} className={arch === a ? 'current' : ''} onClick={() => { choose(a); setModal(null); }}><span className="roadmap-tag"><Icon size={18}/>{t.arch[a].name}</span><span className="roadmap-text"><b>{t.arch[a].long}</b><span>{t.arch[a].description}</span></span></button>; })}</div>
      <h3 className="modal-subtitle">{t.glossary.title}</h3>
      <div className="guide-glossary">{t.glossary.items.map(([term, text]) => <div key={term}><b>{term}</b><span>{text}</span></div>)}</div>
      <p className="fine-print">{arch === 'ann' ? t.ann.fine : arch === 'cnn' ? t.cnn.fine : t.seq.fine} {t.guide.fine}</p>
      <p className="fine-print">{t.app.shortcuts}</p>
      <div className="modal-actions"><button className="primary-button" onClick={() => setModal(null)}>{t.guide.cta} <ArrowRight size={17}/></button><button className="ghost-button" onClick={() => { setModal(null); setTouring(true); }}>{t.app.tourLong}</button></div>
    </Dialog>
  </ShellContext.Provider>;
}
