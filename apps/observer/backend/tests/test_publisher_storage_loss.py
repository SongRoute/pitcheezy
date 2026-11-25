"""OPS-03: what the public publisher does when the artifact drive (T7) goes away.

The real ARM-B policy (``demo_precompute.LivePolicy``) reads its registration and style snapshot from T7 when it
is created, reads the G0 checkpoints from T7 at each game's first bind (and on every rebind), and reads the pinned
``p4_preparation`` JSON from T7 on EVERY call (``history_store``). These tests model exactly that with a FAKE
artifact root in a temp dir that is renamed away to simulate the unmount (the real T7 is never touched):

* lost mid-run: the publisher keeps publishing the game; each new pitch state shows ``compute_failed``
  ("추천 없음"); when the folder is back, the next new state is recommended again (no restart needed);
* (re)started while it is missing: the publisher starts (no crash, so no restart loop), every pitch shows
  ``recommender_unavailable`` ("추천 준비 안 됨"), and that stays so after the drive returns until the publisher
  is restarted (the policy load is attempted once per process).
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer_app.live_armb import LiveArmB
from publish_fixture import World, snap


class DrivePolicy:
    """Wraps the fixture policy: reads a pinned file under ``root`` on every call, as ``history_store`` does."""

    def __init__(self, root, inner):
        self.root, self.inner = Path(root), inner

    def __call__(self, snapshot):
        json.loads((self.root/'p4_preparation.json').read_text())  # FileNotFoundError once the drive is gone
        return self.inner(snapshot)


def drive_factory(root, inner):
    def factory():
        json.loads((Path(root)/'registration.json').read_text())  # LivePolicy.__init__ reads its pins from T7
        return DrivePolicy(root, inner)
    return factory


def fake_drive(tmp_path):
    root = tmp_path/'fake-T7'/'pitchmdp'
    root.mkdir(parents=True)
    for name in ('registration.json', 'p4_preparation.json'):
        (root/name).write_text('{}')
    return root


def unmount(root):
    gone = root.with_name(root.name+'.unmounted')
    root.rename(gone)
    return gone


def state(body):
    return (body['display_state']['state'], body['display_state']['reason'], body['display_state']['recommendation_shown'])


def candidates(body):
    return [c['pitch_type'] for c in ((body.get('next') or {}).get('rec') or {}).get('candidates') or []]


def test_drive_lost_mid_game_keeps_publishing_and_recovers_by_itself(tmp_path):
    root = fake_drive(tmp_path)
    world = World(tmp_path/'w')
    try:
        world.armb = LiveArmB(drive_factory(root, world.policy))
        world.start()
        world.cycle()
        assert candidates(world.game()) == ['SL']
        first = world.read('v1/health.json')['sequence']

        gone = unmount(root)
        world.cycle(snap('20261005_100100', call='C'))  # a new pitch state: computed while the drive is away
        body = world.game()
        assert state(body) == ('unavailable', 'compute_failed', False) and candidates(body) == []
        health = world.read('v1/health.json')
        assert health['sequence'] > first and health['unavailable_games'] == []  # the game itself is still published
        assert world.armb.status()['state'] == 'ready'  # the loaded policy stays loaded: nothing restarts

        gone.rename(root)  # reconnected at the same mount point
        world.cycle(snap('20261005_100200', call='B', second=True))
        assert candidates(world.game()) == ['SL']
    finally:
        world.close()


def test_publisher_started_without_the_drive_runs_without_recommendations_until_restarted(tmp_path):
    root = fake_drive(tmp_path)
    gone = unmount(root)
    world = World(tmp_path/'w')
    try:
        world.armb = LiveArmB(drive_factory(root, world.policy))
        world.start()  # build() never reads the drive: no crash, so the supervisor sees no exit and no restart loop
        world.cycle()
        body = world.game()
        assert state(body) == ('unavailable', 'recommender_unavailable', False) and candidates(body) == []
        assert world.armb.status()['state'] == 'unavailable'
        assert world.read('v1/health.json')['unavailable_games'] == []  # nothing in health names it

        gone.rename(root)  # back, but this process already gave up on the policy
        world.cycle(snap('20261005_100100', call='C'))
        assert state(world.game())[1] == 'recommender_unavailable'

        world.armb = LiveArmB(drive_factory(root, world.policy))  # a publisher restart (new process, new policy)
        world.start()
        world.cycle(snap('20261005_100200', call='B', second=True))
        assert candidates(world.game()) == ['SL']
    finally:
        world.close()
