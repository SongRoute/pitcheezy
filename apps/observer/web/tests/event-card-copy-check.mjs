// Render the actual legacy EventCard in isolation; the current service entry does not bundle App.tsx.
// Synthetic contract cases only. Run: node tests/event-card-copy-check.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
const source = ts.createSourceFile('App.tsx', readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const names = ['EventCard', 'finite', 'signed'];
const selected = source.statements.filter(s => names.includes(s.name?.text) || s.declarationList?.declarations.some(d => names.includes(d.name.text)));
assert.equal(selected.length, names.length);
const code = ts.transpileModule(selected.map(s => s.getText(source)).join('\n'), { compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2022 } }).outputText;
const EventCard = new Function('React', `${code}; return EventCard;`)(React);
for (const reason of ['missing_intent', 'missing_compatible_values']) {
  const event = { status: 'partial', reason, values: { total_pp: null },
    components: {}, evidence: { actual_source: 'historical_replay_record',
      intent_source: reason === 'missing_intent' ? null : 'reviewed_video', development_only: false } };
  const html = renderToStaticMarkup(React.createElement(EventCard, { event }));
  assert.match(html, /비교에 필요한 근거가 부족해 일부 항목만 계산할 수 있습니다/);
  assert.doesNotMatch(html, /투구 의도 정보가 없어|미측정: 투구 의도/);
  assert.match(html, /남은 차이를 선수에게 배분하지 않습니다/);
  assert.match(html, /선수별 인과적 기여/);
  assert.match(html, reason === 'missing_intent' ? /투구 전 의도 근거 없음/ : /비교에 필요한 값 부족/);
  if (event.evidence.intent_source) assert.match(html, /영상 입력 근거 있음/);
}
console.log('2 synthetic partial cases passed');
