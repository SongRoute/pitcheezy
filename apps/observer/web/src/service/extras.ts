/** Optional additions of the live fan payload (backend service_extras.py; each behind an owner switch, default off).
 *  Payload-driven: a field that is absent or not in the expected shape is simply not shown. Nothing here is a recommendation.
 *  - `next.reference`: past pitch frequencies of a pitcher the recommender does not cover (A2), shown beside the unchanged
 *    "추천 없음" state as a labelled reference.
 *  - `pitches[i].expectation`: one quiet sentence in the detail sheet of a past key-moment pitch (B2). */
type Raw = Record<string, any>;
export type ReferenceItem = { pitch_type: string; label: string; pct: number };
/** `stale_label`: "기록 기준 MM/DD", only when the server marks the card stale (its profile cutoff is more than 2 days before the game
 *  date: daily refreshes were missed), so an old record is dated up front, not only in the sheet. */
export type ReferenceCard = { chip: string; title: string; items: ReferenceItem[]; other_pct: number | null; sample: string | null; note: string; stale_label: string | null };
export type Expectation = { text: string; note: string | null; explain: string | null };

const str = (v: unknown): string | null => typeof v === 'string' && v.trim() ? v.trim() : null;
const whole = (v: unknown): number | null => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 100 ? Math.round(v) : null;
const NOT_A_PICK = '추천이나 다음 공 예측이 아닙니다.';

/** What the reference is, in this screen's own words (a payload cannot relabel it). */
export const REFERENCE_CHIP = '참고 · 기록 통계';
/** Pitcher-level reasons of the unsupported state (backend service_extras.PITCHER_REASONS). */
const PITCHER_REASONS = ['unknown_pitcher', 'no_supported_pitch_types'];
/** `next.reference` → card. Shown only where the server attaches it: the profile-reference kind with its chip, a title and at least
 *  one item, beside a pending pitch (`pre_pitch`) whose state is unsupported for a pitcher-level reason. Anything else → null. */
export function normalizeReference(raw: Raw | null | undefined): ReferenceCard | null {
  const card = raw?.next?.reference, state = raw?.display_state;
  if (!card || typeof card !== 'object' || card.kind !== 'profile_reference' || !str(card.chip) || !Array.isArray(card.items)) return null;
  if (raw?.current?.phase !== 'pre_pitch' || state?.state !== 'unsupported' || !PITCHER_REASONS.includes(state?.reason)) return null;
  const items: ReferenceItem[] = card.items.flatMap((i: Raw) => { const label = str(i?.pitch_label) ?? str(i?.pitch_type); const pct = whole(i?.share_pct);
    return label && pct !== null ? [{ pitch_type: String(i.pitch_type ?? label), label, pct }] : []; });
  const title = str(card.title);
  if (!title || !items.length) return null;
  // the date as the server states it, in full: no "올해" is assumed here (the server's own note says which season it is)
  const day = /^(\d{4})-(\d{2})-(\d{2})$/.exec(card.as_of ?? '');
  const n = typeof card.sample?.pitches === 'number' ? card.sample.pitches : null;
  return { chip: REFERENCE_CHIP, title, items, other_pct: whole(card.other?.share_pct),
    sample: day && n !== null ? `${Number(day[1])}년 ${Number(day[2])}월 ${Number(day[3])}일까지 같은 상황 ${n}구` : n !== null ? `같은 상황 ${n}구` : null,
    note: str(card.note) ?? NOT_A_PICK, stale_label: card.stale === true && day ? `기록 기준 ${day[2]}/${day[3]}` : null };
}
/** `pitches[i].expectation` → sentence; null unless it is the step-5 kind with a text. */
export function normalizeExpectation(raw: Raw | null | undefined): Expectation | null {
  const e = raw?.expectation;
  if (!e || typeof e !== 'object' || e.kind !== 'step5_expectation' || !str(e.text)) return null;
  return { text: str(e.text)!, note: str(e.note), explain: str(e.explain) };
}
export const referenceLine = (card: ReferenceCard) => card.items.map(i => `${i.label} ${i.pct}%`).join(' · ');
