import type { Expectation, ReferenceCard } from './extras';
import { referenceLine } from './extras';
import { Sheet } from './Sheets';

/** A2. One line inside the (fixed-height) state slot, under the unchanged "추천 없음" sentence: the chip that says what this is and
 *  the frequencies in words. It opens the full card. No rank, no zone mark, none of the recommendation colours. */
export function ReferenceTeaser({ card, onOpen }: { card: ReferenceCard; onOpen: () => void }) {
  return <button className="sv-ref-teaser" onClick={onOpen} aria-haspopup="dialog"
    aria-label={`${card.chip}. ${card.title}: ${referenceLine(card)}. 추천이 아니에요. 자세히 보기`}>
    <span className="sv-ref-chip" aria-hidden="true">{card.chip}</span><span className="sv-ref-line" aria-hidden="true">{referenceLine(card)}</span><i aria-hidden="true">›</i></button>;
}

/** A2. The whole reference card: chip, title, "구종 NN%" rows (past frequency), the sample line and the note that it is neither a
 *  recommendation nor a prediction of the next pitch. */
export function ReferenceSheet({ card, state, onClose }: { card: ReferenceCard; state: string | null; onClose: () => void }) {
  return <Sheet title="참고 기록" onClose={onClose}>
    {state && <p className="sv-muted sv-ref-state">{state}</p>}
    <section className="sv-ref-card" aria-label={card.chip}>
      <span className="sv-ref-chip">{card.chip}</span>
      <h3>{card.title}</h3>
      <ul>{card.items.map(i => <li key={i.pitch_type}><span>{i.label}</span><b>{i.pct}%</b></li>)}
        {card.other_pct !== null && card.other_pct > 0 && <li className="other"><span>그 밖의 공</span><b>{card.other_pct}%</b></li>}</ul>
      {card.sample && <p className="sv-ref-sample">{card.sample}</p>}
      <p className="sv-ref-note">{card.note}</p>
    </section>
  </Sheet>;
}

/** B2. One quiet line in the realized-pitch part of the detail sheet of a past key-moment pitch. */
export function ExpectationLine({ e }: { e: Expectation }) {
  return <div className="sv-expect"><p className="sv-expect-text">{e.text}{e.note && <small> · {e.note}</small>}</p>
    {e.explain && <p className="sv-muted sv-expect-explain">{e.explain}</p>}</div>;
}
