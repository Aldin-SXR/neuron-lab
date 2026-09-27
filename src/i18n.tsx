import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { format } from './engine';
import { en, type Messages } from './locales/en';
import { bs } from './locales/bs';

export type Lang = 'en' | 'bs';
export const LANGUAGES: Record<Lang, Messages> = { en, bs };
const STORAGE_KEY = 'neuron-lab-lang';

function initialLang(): Lang {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'en' || stored === 'bs') return stored;
  } catch { /* Storage may be unavailable; fall back to the browser language. */ }
  return /^(bs|hr|sr|sh)\b/i.test(navigator.language) ? 'bs' : 'en';
}
interface I18n {
  lang: Lang; t: Messages; setLang: (lang: Lang) => void;
  /** A number for prose and metrics, with the language's decimal separator (formulas keep the point). */
  num: (value: number, digits?: number) => string;
  /** Separator between numbers in a list: a comma already serves as the Bosnian decimal separator. */
  listSep: string;
}
const localized = (lang: Lang) => ({
  num: (value: number, digits?: number) => lang === 'bs' ? format(value, digits).replace('.', ',') : format(value, digits),
  listSep: lang === 'bs' ? '; ' : ', ',
});
const I18nContext = createContext<I18n>({ lang: 'en', t: en, setLang: () => {}, ...localized('en') });

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(initialLang);
  useEffect(() => {
    document.documentElement.lang = lang;
    document.title = LANGUAGES[lang].app.documentTitle;
    try { localStorage.setItem(STORAGE_KEY, lang); } catch { /* The choice still applies for this visit. */ }
  }, [lang]);
  return <I18nContext.Provider value={{ lang, t: LANGUAGES[lang], setLang, ...localized(lang) }}>{children}</I18nContext.Provider>;
}
export const useI18n = () => useContext(I18nContext);
