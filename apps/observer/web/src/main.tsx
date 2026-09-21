import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import VideoLab from './VideoLab';
import './styles.css';

createRoot(document.getElementById('root')!).render(<React.StrictMode>{window.location.pathname.replace(/\/$/, '') === '/video-lab' ? <VideoLab /> : <App />}</React.StrictMode>);
