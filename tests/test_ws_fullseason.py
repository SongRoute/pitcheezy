"""Generated data only; never fixture files or real registrations/payloads."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from experiments.ws_fullseason import data

RECIPE = Path(__file__).resolve().parents[1] / 'configs/EXP-WS-FULLSEASON-T-001.json'


def rows():
    result = []
    for game, day in enumerate(['2026-03-25', '2026-06-30', '2026-07-01', '2026-07-31'], 900000):
        for pitch, description in enumerate(['ball', 'called_strike', 'foul'], 1):
            result.append(dict(game_pk=game, game_date=day, game_type='R', at_bat_number=1,
                pitch_number=pitch, home_team='AAA', away_team='BBB', pitcher=101, batter=202,
                pitch_type='FF', description=description, events=None, bb_type=None,
                release_speed=90., release_spin_rate=2100., pfx_x=.2, pfx_z=1.,
                balls=0, strikes=0, outs_when_up=0, inning=1, inning_topbot='Top',
                home_score=0, away_score=0, on_1b=None, on_2b=None, on_3b=None, stand='R', p_throws='R'))
    return pd.DataFrame(result)


class Quality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = data.closure()
        cls.recipe = json.loads(RECIPE.read_text())

    def test_exact_role_key_accounting(self):
        g, census, fit, oof, y = data.project(rows(), self.recipe, self.M)
        report = data.audit_summary(g, census, fit, oof, y, self.M)
        self.assertEqual((fit.sum(), oof.sum(), census['accepted_rows']), (6, 6, 12))
        self.assertEqual(sum(report['roles']['OOF']['seven_leaf_counts']), 6)
        self.assertNotEqual(report['roles']['FIT']['ordered_keys_sha256'], report['roles']['OOF']['ordered_keys_sha256'])

    def test_quality_exclusion_stops(self):
        f = rows(); f.loc[0, 'description'] = 'unregistered-description'
        with self.assertRaises(self.M.cp.ProjectionStop): data.project(f, self.recipe, self.M)

    def test_conflicting_duplicate_stops(self):
        f = rows(); extra=f.iloc[[0]].copy(); extra['balls']=1
        with self.assertRaisesRegex(ValueError, 'non-identical duplicate'): data.project(pd.concat([f, extra]), self.recipe, self.M)

    def test_empty_oof_stops(self):
        with self.assertRaisesRegex(ValueError, 'empty'): data.project(rows().iloc[:6], self.recipe, self.M)

    def test_future_outcome_does_not_change_fit_projection(self):
        raw=rows(); g,_,fit,_,_=data.project(raw, self.recipe, self.M)
        raw.loc[raw.game_date > '2026-06-30', 'description']='swinging_strike'
        changed,_,other,_,_=data.project(raw, self.recipe, self.M)
        pd.testing.assert_frame_equal(g.loc[fit], changed.loc[other])

    def test_fractional_key_and_postcast_collision_stop(self):
        for value in [1.5, -1, 0, 2**53]:
            f=rows(); f['pitch_number']=f.pitch_number.astype(float); f.loc[0,'pitch_number']=value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'invalid integral identity'):
                data.project(f, self.recipe, self.M)
        f=rows(); f['pitcher']=-3
        with self.assertRaisesRegex(ValueError, 'invalid integral identity'): data.project(f, self.recipe, self.M)

    def test_key_digest_rejects_wrong_shape(self):
        with self.assertRaises(ValueError): data.key_digest(np.zeros((4,2)))


if __name__ == '__main__': unittest.main()
