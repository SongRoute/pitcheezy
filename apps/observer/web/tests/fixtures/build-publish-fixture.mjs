// Builds tests/fixtures/publish/v1/** — the published-file layout of docs/pm/service/public-publish-contract-draft.md (v0) — by
// hand-shaping the existing UI fixture (tests/fixtures/service-849843.json, real 849843 first 65 pitches). No network, no real feed.
// Each file is the service answer plus the publisher's one added top-level key, as on branch svc/ws-publisher
// (apps/observer/publish_contract/v1): "publish": { contract, sequence, published_at[, final] }. health.json and manifest.json follow
// that branch's examples. Synthetic parts are marked `synthetic`. Sequences, times and intervals here are fixture values, not measurements.
//   node tests/fixtures/build-publish-fixture.mjs
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(path.join(here, 'service-849843.json'), 'utf8'));
const out = path.join(here, 'publish', 'v1');
const SEQ = 120, AT = '2026-09-30T05:00:00Z';
const stamp = (body, seq = SEQ, at = AT, final) => ({ ...body, publish: { contract: 'v1', sequence: seq, published_at: at, ...(final === undefined ? {} : { final }) } });
const write = (file, body) => { const p = path.join(out, file); mkdirSync(path.dirname(p), { recursive: true }); writeFileSync(p, JSON.stringify(body)); console.log(file, Buffer.byteLength(JSON.stringify(body)), 'bytes'); };

// finished game: the archive answer as it is
const final = structuredClone(fx.final);
// synthetic live game 990003: the fixture's 31 real pitches marked as the live route's answer with a display state that shows the list
const live = structuredClone(fx.live);
live.synthetic = 'game_pk, teams on the card, source.kind and display_state are synthetic; pitches are real 849843 pitches';
live.source = { kind: 'live', session_id: null };
live.game = { ...live.game, game_pk: 990003, status: 'live', status_text: 'In Progress', live: true };
live.receipts = { version: 'fixture', corrected: 0, paired: 0, unconfirmed: 0 };
live.display_state = { schema: 'pitcheezy-display-state-v1', state: 'recommended', reason: 'ledger_recorded', label: '다음 공 추천', message: null,
  pitch_key: live.next.pitch_key, recommendation_shown: true, correction: null, feed: { ok: true, delayed: false, lag_s: null } };

const cards = [final.game, live.game];
write('dates.json', stamp({ schema: 'pitcheezy-service-dates-v2', source: { kind: 'archive', session_id: null }, dates: ['2026-09-30'] }));
write('games/2026-09-30.json', stamp({ schema: 'pitcheezy-service-games-v2', source: { kind: 'archive', session_id: null }, date: '2026-09-30', games: cards, live_ok: true }));
write('games/2026-09-29.json', stamp({ schema: 'pitcheezy-service-games-v2', source: { kind: 'archive', session_id: null }, date: '2026-09-29', games: [], live_ok: true }));
// a settled final game keeps the sequence of the cycle that last wrote it, far behind health.json
write('game/849843.json', stamp(final, 3, '2026-09-30T04:40:30Z', true));
write('game/990003.json', stamp(live, SEQ, AT, false));
write('manifest.json', stamp({ schema: 'pitcheezy-publish-manifest-v1', contract: 'v1', commit: '0000000000000000000000000000000000000000', commit_dirty: false, published_at: AT,
  publish_interval_s: 10, fixed_delay_s: 0, provisional: ['publish_interval_s', 'settled_recheck_s', 'live_days_back', 'live_days_ahead'], live_dates: ['2026-09-29', '2026-09-30'] }));
write('health.json', stamp({ schema: 'pitcheezy-publish-health-v1', sequence: SEQ, published_at: AT, publish_interval_s: 10, files_written: 6, unavailable_games: [], unavailable_dates: [], feed_failing_games: [] }));
