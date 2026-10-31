"""Makes the trimmed 849845 live fixtures look like a real feed where the store's admission rule needs it."""
import gzip
import json
from pathlib import Path

FIXTURES = Path(__file__).parent/'fixtures/live'


def listed(snapshot):
    """The registered fixtures carry ``currentPlay`` only: no ``allPlays`` and no provider ``playId``. A real feed
    lists the current play as its last ``allPlays`` entry and identifies every pitch, which is what the pitch
    ledger admits; a ``currentPlay`` pitch the ledger never saw is (rightly) not shown as an observed pitch.
    Returns ``snapshot`` with the play listed and its pitches identified."""
    plays = snapshot['liveData']['plays']
    play = plays.get('currentPlay')
    if play and not plays.get('allPlays'):
        for event in play.get('playEvents') or []:
            if event.get('isPitch'):
                event.setdefault('playId', f"fixture-{play['about']['atBatIndex']}-{event['pitchNumber']}")
        plays['allPlays'] = [play]
    return snapshot


def listed_dir(target):
    """A replay folder (``ReplayTransport``) holding the fixtures as ``listed`` feeds; returns ``target``."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(FIXTURES.glob('*.json')):
        body = json.loads(path.read_text())
        if 'liveData' in body:
            body = listed(body)
        with gzip.open(target/(path.name+'.gz'), 'wt') as handle:
            json.dump(body, handle)
    return target
