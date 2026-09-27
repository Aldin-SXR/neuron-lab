import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { DivergedError, type LabFrame } from './lab';

export type Mode = 'learn' | 'train';
export interface Timeline<F> { past: F[]; present: F; future: F[]; error?: string }
interface Operations<F> { step: (frame: F, rate: number) => F; epoch: (frame: F, rate: number) => F }
const HISTORY = 500;
/** Parameters that blow up are an expected teaching moment; anything else is a real bug worth logging. */
function failure(error: unknown) {
  if (error instanceof DivergedError) return 'diverged';
  console.error(error);
  return 'unexpected';
}

/**
 * Undoable playback shared by every lab: stepping, training epochs, autoplay,
 * and keyboard shortcuts (← back, → next, space play/pause) while the lab is active.
 */
export function useTimeline<F extends LabFrame>(initial: () => F, operations: Operations<F>, rate: number, active: boolean) {
  const [timeline, setTimeline] = useState<Timeline<F>>(() => ({ past: [], present: initial(), future: [] }));
  const [mode, setMode] = useState<Mode>('learn');
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const ops = useRef(operations);
  useLayoutEffect(() => { ops.current = operations; });

  const advance = useCallback(() => {
    setTimeline(t => {
      try {
        const next = mode === 'learn' ? ops.current.step(t.present, rate) : ops.current.epoch(t.present, rate);
        return { past: [...t.past, t.present].slice(-HISTORY), present: next, future: [] };
      } catch (error) { return { ...t, error: failure(error) }; }
    });
  }, [mode, rate]);
  useEffect(() => {
    if (!playing || timeline.error || !active) return;
    const interval = window.setInterval(advance, mode === 'learn' ? 950 / speed : 80 / speed);
    return () => window.clearInterval(interval);
  }, [advance, playing, mode, speed, timeline.error, active]);
  useEffect(() => { if (!active) setPlaying(false); }, [active]);

  const back = useCallback(() => {
    setPlaying(false);
    setTimeline(t => t.past.length ? { past: t.past.slice(0, -1), present: t.past.at(-1)!, future: [t.present, ...t.future] } : t);
  }, []);
  const next = useCallback(() => {
    setPlaying(false);
    setTimeline(t => {
      if (t.future.length) return { past: [...t.past, t.present].slice(-HISTORY), present: t.future[0], future: t.future.slice(1) };
      if (t.error) return t;
      try {
        const frame = mode === 'learn' ? ops.current.step(t.present, rate) : ops.current.epoch(t.present, rate);
        return { past: [...t.past, t.present].slice(-HISTORY), present: frame, future: [] };
      } catch (error) { return { ...t, error: failure(error) }; }
    });
  }, [mode, rate]);
  const togglePlay = useCallback(() => setPlaying(v => !v), []);
  useEffect(() => {
    if (!active) return;
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || target.closest('input, select, textarea, dialog, [contenteditable]')) return;
      if (e.key === 'ArrowRight') { e.preventDefault(); next(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); back(); }
      // Space already activates a focused button, so only handle it elsewhere.
      else if (e.key === ' ' && !target.closest('button, a, [role=button]')) { e.preventDefault(); togglePlay(); }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active, next, back, togglePlay]);

  function changeMode(nextMode: Mode) { setMode(nextMode); setPlaying(false); setTimeline(t => ({ ...t, future: [] })); }
  /** Start a new history from this frame (architecture, data, or restored weights changed). */
  function reset(frame: F) { setPlaying(false); setTimeline({ past: [], present: frame, future: [] }); }
  /** Record a new state that can be undone (for example, selecting another sample). */
  function push(update: (frame: F) => F) { setPlaying(false); setTimeline(t => ({ past: [...t.past, t.present].slice(-HISTORY), present: update(t.present), future: [] })); }
  function clearFuture() { setTimeline(t => ({ ...t, future: [] })); }
  const frame = timeline.present;
  const lockedRate = frame.cursor > 0 && frame.cursor < frame.plan.length - 1;
  return { frame, timeline, mode, changeMode, playing, setPlaying, togglePlay, speed, setSpeed, back, next, reset, push, clearFuture, lockedRate, canGoBack: timeline.past.length > 0 };
}
/** The subset of the timeline that playback controls need, independent of the frame type. */
export interface PlaybackControls { frame: LabFrame; mode: Mode; playing: boolean; timeline: { error?: string }; speed: number; setSpeed: (v: number) => void; back: () => void; next: () => void; togglePlay: () => void; canGoBack: boolean }
