import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import VideoLab from './VideoLab';
import './styles.css';

function render() {
  createRoot(document.getElementById('root')!).render(<React.StrictMode>{window.location.pathname.replace(/\/$/, '') === '/video-lab' ? <VideoLab /> : <App />}</React.StrictMode>);
}

// VITE_API_MOCK=1 (Vercel 프리뷰·npm run dev:mock): 백엔드 없이 /api 를 MSW fixture 로 응답한다. docs/ui/WORKFLOW.md
if (import.meta.env.VITE_API_MOCK === '1') {
  import('./mocks/browser').then(({ startMockApi }) => startMockApi()).catch((error) => { console.error('mock API 시작 실패', error); }).finally(render);
} else {
  render();
}
