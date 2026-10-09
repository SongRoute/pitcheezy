const app = document.querySelector('#app');
let current = null;
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const json = value => escapeHTML(JSON.stringify(value, null, 1));

async function request(path, method = 'GET') {
  const response = await fetch(path, { method, credentials: 'same-origin', cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '불러오지 못했습니다.');
  return data;
}

function historyHTML(rows) {
  if (!rows.length) return '<p class="empty">공을 공개하면 이곳에 받은 투구 기록이 쌓입니다.</p>';
  return `<ol class="history-list">${rows.slice().reverse().map(row => `<li><span>${row.game_pitch}구 · ${row.inning}회</span><b>${escapeHTML(row.result)}${row.corrected ? ' <small>정정됨</small>' : ''}${row.changed_from ? ` <small>표시 바뀜 · 이전 ${escapeHTML(row.changed_from)}</small>` : ''}</b></li>`).join('')}</ol>`;
}

function render() {
  const view = current;
  const s = view.situation;
  const c = view.comparison;
  const a = view.actual;
  app.innerHTML = `<section class="hero">
    <div><div class="eyebrow">SYNTHETIC · ${escapeHTML(view.game_label)} · ${view.game_pitch}번째 투구</div>
      <h1>${s ? `${s.inning}회${escapeHTML(s.half)}` : '경기 시작 전'}</h1>
      <p>${s ? `타자 #${s.batter} · 투수 #${s.pitcher} · 마지막으로 받은 자료 기준` : '아직 받은 자료가 없습니다.'}</p>
      ${view.connection ? `<p class="status" role="status">${escapeHTML(view.connection)}</p>` : ''}</div>
    <div class="scoreboard" aria-label="받은 자료 기준 카운트"><div><strong>${s ? s.count.balls : '-'}</strong><small>볼</small></div><div><strong>${s ? s.count.strikes : '-'}</strong><small>스트라이크</small></div><div><strong>${view.game_pitch}</strong><small>투구</small></div></div>
  </section>
  <section class="layout">
    <div class="card"><div class="section-head"><div><div class="number">PITCH ${view.game_pitch} · ${view.reveal ? '공개 후' : '투구 전'}</div><h2>${view.reveal ? '받은 투구 결과' : '이번 공 비교'}</h2></div><span class="tag">${view.reveal ? '공개됨' : '공개 전'}</span></div>
      ${c ? `<div class="state state-${escapeHTML(c.state)}"><strong>${escapeHTML(c.label)}</strong><p>${escapeHTML(c.reason)}</p></div>` : ''}
      ${a ? `<div class="actual"><small>관측기가 받은 기록</small><strong>${escapeHTML(a.result ?? '자료 없음')}</strong>${a.note ? `<p>${escapeHTML(a.note)}</p>` : ''}</div>` : ''}
      <div class="action-bar">${!view.reveal ? '<button type="button" class="primary" id="reveal">결과 공개</button>' : view.complete ? '<button type="button" class="secondary" id="restart">처음부터 다시 보기</button>' : '<button type="button" class="primary" id="next">다음 투구</button>'}</div>
      ${view.complete ? `<div class="summary"><b>합성 경기 종료 · 기대 키 ${view.partition.expected_keys}개</b><p>수용 ${view.partition.counts.accepted} · 격리만 ${view.partition.counts.quarantined_only} · 누락 ${view.partition.counts.missing} · 계획 밖 ${view.partition.counts.extra}</p><small>${escapeHTML(view.partition.note)}</small></div>` : ''}
      <details class="detail diag"><summary>근거/진단</summary><pre>${json(view.diagnostics)}${c ? '\n' + json(c.gate) : ''}${a?.reason ? '\n' + json({display_reason: a.reason}) : ''}</pre></details>
    </div>
    <aside class="card"><div class="section-head"><div><div class="number">PITCH LOG</div><h2>지금까지의 투구</h2><p>공개된 기록만 표시합니다.</p></div></div>${historyHTML(view.history)}
      <p class="note">${view.limitations.map(escapeHTML).join('<br>')}</p></aside>
  </section>`;
  document.querySelector('#reveal')?.addEventListener('click', () => transition('/api/reveal'));
  document.querySelector('#next')?.addEventListener('click', () => transition('/api/next'));
  document.querySelector('#restart')?.addEventListener('click', () => transition('/api/restart'));
}

async function transition(path) {
  try { current = await request(path, 'POST'); render(); }
  catch (error) {
    app.insertAdjacentHTML('afterbegin', `<p class="error" role="alert">${escapeHTML(error.message)} <button type="button" id="retry">다시 불러오기</button></p>`);
    document.querySelector('#retry')?.addEventListener('click', start);
  }
}

async function start() {
  try { current = await request('/api/state'); render(); }
  catch (error) { app.innerHTML = `<p class="error" role="alert">${escapeHTML(error.message)} <button type="button" id="retry">다시 시도</button></p>`; document.querySelector('#retry').addEventListener('click', start); }
}
start();
