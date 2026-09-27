import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, X } from 'lucide-react';
import { useI18n } from './i18n';

const STEPS = ['architecture', 'problem', 'network', 'playback', 'mode', 'inspector', 'results', 'settings', 'language'];
function target(id: string) {
  return document.querySelector<HTMLElement>(`.lab:not([hidden]) [data-tour="${id}"]`) ?? document.querySelector<HTMLElement>(`header [data-tour="${id}"]`);
}
/** A lightweight coach-mark tour: dims the page, spotlights one element, and explains it. */
export function Tour({ onClose }: { onClose: () => void }) {
  const { t } = useI18n();
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const card = useRef<HTMLDivElement>(null);
  const id = STEPS[index];
  const measure = useCallback(() => { const el = target(id); setRect(el ? el.getBoundingClientRect() : null); }, [id]);
  useLayoutEffect(() => {
    const el = target(id);
    el?.scrollIntoView({ block: 'center', inline: 'nearest' });
    measure();
    card.current?.focus();
  }, [id, measure]);
  useEffect(() => {
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', measure, true);
    return () => { window.removeEventListener('resize', measure); window.removeEventListener('scroll', measure, true); };
  }, [measure]);
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); if (index < STEPS.length - 1) setIndex(index + 1); else onClose(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); setIndex(Math.max(0, index - 1)); }
      else return;
      e.stopPropagation();
    }
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [index, onClose]);
  const [title, body] = t.tour.steps[id];
  const pad = 8;
  const narrow = window.innerWidth < 640;
  const width = Math.min(360, window.innerWidth - 32);
  let style: React.CSSProperties = { width };
  if (rect && !narrow) {
    const below = rect.bottom + pad + 16, spaceBelow = window.innerHeight - below;
    const left = Math.min(Math.max(16, rect.left + rect.width / 2 - width / 2), window.innerWidth - width - 16);
    style = spaceBelow > 230 ? { width, left, top: below } : rect.top > 250 ? { width, left, bottom: window.innerHeight - rect.top + pad + 16 } : { width, left: Math.min(rect.right + 16, window.innerWidth - width - 16), top: Math.max(16, rect.top) };
  }
  return <div className="tour-layer">
    <div className="tour-backdrop" onClick={e => e.stopPropagation()}/>
    {rect && <div className="tour-spotlight" style={{ left: rect.left - pad, top: rect.top - pad, width: rect.width + pad * 2, height: rect.height + pad * 2 }}/>}
    <div className={`tour-card ${narrow || !rect ? 'docked' : ''}`} style={style} role="dialog" aria-modal="true" aria-labelledby="tour-title" tabIndex={-1} ref={card}>
      <div className="tour-top"><span className="eyebrow">{t.tour.progress(index + 1, STEPS.length)}</span><button className="icon-button" onClick={onClose} aria-label={t.tour.skip} title={t.tour.skip}><X size={18}/></button></div>
      <h3 id="tour-title">{title}</h3><p>{body}</p>
      <div className="tour-dots" aria-hidden>{STEPS.map((s, i) => <span key={s} className={i === index ? 'active' : ''}/>)}</div>
      <div className="tour-actions">
        <button className="text-button" onClick={onClose}>{t.tour.skip}</button>
        <span>
          {index > 0 && <button className="ghost-button" onClick={() => setIndex(index - 1)}><ArrowLeft size={15}/> {t.tour.back}</button>}
          <button className="primary-button" onClick={() => index < STEPS.length - 1 ? setIndex(index + 1) : onClose()}>{index < STEPS.length - 1 ? t.tour.next : t.tour.done} <ArrowRight size={15}/></button>
        </span>
      </div>
    </div>
  </div>;
}
