import type { Game } from './types';

export type Bookmarks = Record<string, string>;
const BOOKMARK_KEY = 'pitcheezy.observer.bookmarks.v1';
export const bookmarkKey = (gameId: number, paId: number) => `${gameId}:${paId}`;
const normalize = (value: string) => value.normalize('NFKC').toLocaleLowerCase().replace(/[,·]/g, ' ');

/** Explicit metadata allowlist: never search serialized records or outcome fields. */
export function filterCatalog(games: Game[], query: string, date: string): Game[] {
  const tokens = normalize(query).trim().split(/\s+/).filter(Boolean);
  return games.filter((game) => !date || game.date === date).map((game) => {
    const gameText = [game.home_team, game.away_team, game.title].join(' ');
    const plate_appearances = (game.plate_appearances || []).filter((pa) => {
      const text = normalize([gameText, pa.batter_label, pa.pitcher_label].join(' '));
      return tokens.every((token) => text.includes(token));
    });
    return { ...game, plate_appearances };
  }).filter((game) => game.plate_appearances.length > 0);
}

export function readBookmarks(): Bookmarks {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(BOOKMARK_KEY) || '{}');
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter(([key, id]) => /^\d+:\d+$/.test(key) && typeof id === 'string' && id.length < 128));
  } catch { return {}; }
}

export function rememberSession(gameId: number, paId: number, id: string): Bookmarks {
  const bookmarks = readBookmarks();
  const key = bookmarkKey(gameId, paId);
  delete bookmarks[key];
  bookmarks[key] = id;
  const recent = Object.fromEntries(Object.entries(bookmarks).slice(-60));
  try { localStorage.setItem(BOOKMARK_KEY, JSON.stringify(recent)); } catch { /* A current session still works without storage. */ }
  return recent;
}
