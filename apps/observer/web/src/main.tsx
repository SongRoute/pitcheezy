import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import VideoLab from './VideoLab';
import InningDecisions from './InningDecisions';
import './styles.css';

const route = window.location.pathname.replace(/\/$/, '');
createRoot(document.getElementById('root')!).render(<React.StrictMode>{route === '/video-lab' ? <VideoLab /> : route === '/inning-decisions' ? <InningDecisions /> : <App />}</React.StrictMode>);
