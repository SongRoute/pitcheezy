import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import VideoLab from './VideoLab';
import InningDecisions from './InningDecisions';
import WatchAlong from './WatchAlong';
import Live from './Live';
import Game from './Game';
import './styles.css';

const route = window.location.pathname.replace(/\/$/, '');
createRoot(document.getElementById('root')!).render(<React.StrictMode>{route === '/video-lab' ? <VideoLab /> : route === '/inning-decisions' ? <InningDecisions /> : route === '/watch' ? <WatchAlong /> : route === '/live' ? <Live /> : route === '/game' ? <Game /> : <App />}</React.StrictMode>);
