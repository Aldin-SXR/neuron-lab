import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { I18nProvider } from './i18n';
import '@fontsource-variable/dm-sans';
import '@fontsource-variable/manrope';
import './styles.css';
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><I18nProvider><App /></I18nProvider></React.StrictMode>);
