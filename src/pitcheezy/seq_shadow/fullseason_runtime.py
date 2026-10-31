"""Isolated seven-leaf shadow composition; no observer, WE, collector or host I/O."""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from . import fullseason, fullseason_profiles as profiles, fullseason_stream as stream
from .ledger import sha


class Runtime:
    """Load one exact model/encoder/calibrator/profile identity; no implicit reload.

    Actual artifact reads require the external registered grant. Retrospective
    bundle/model development status is preserved even for later captured inputs.
    Each call is bounded to one query and 32 accepted prior tokens, no raw donors.
    """
    def __init__(self, adapter_config, profile_directory, profile_manifest_sha256):
        config = copy.deepcopy(adapter_config)
        self.ensemble = fullseason.load(config)
        self.profiles = profiles.load_bundle(profile_directory, profile_manifest_sha256)
        self.identity = {
            'adapter_config_sha256': sha(config),
            'profile_manifest_sha256': profile_manifest_sha256,
            'semantics': 'conditional_seven_leaf_shadow; mean_three_raw_seeds_then_single_temperature',
            'role': '2026 EXPOSED DEVELOPMENT',
            'profile_arrival': 'UNKNOWN; corrected retrospective statistics',
        }
        self.bundle_sha256 = sha(self.identity)

    def predict(self, snapshot):
        # Verify snapshot before using its identity or abstention; no altered
        # provenance may inherit an earlier decision's cache/receipt identity.
        body = dict(snapshot); digest = body.pop('snapshot_sha256')
        if sha(body) != digest: raise ValueError('snapshot changed')
        result = {'status': snapshot['status'], 'reason': snapshot['reason'],
            'bundle_sha256': self.bundle_sha256, 'snapshot_sha256': digest,
            'call_sha256': sha({'bundle': self.bundle_sha256, 'snapshot': digest}),
            'pitch_type_basis': snapshot['pitch_type_basis'],
            'stream_evidence_kind': snapshot['identity']['evidence_kind'],
            'prospective_claim': False, 'leaf_order': list(fullseason.LEAVES),
            'probabilities': None, 'raw_mean': None}
        if snapshot['status'] != 'ready': return result
        tokens = stream.arrays(snapshot)
        query = pd.DataFrame([snapshot['query']], columns=stream.SCHEMA)
        # None means explicitly known-empty, not unknown base occupancy.
        for column in ('on_1b', 'on_2b', 'on_3b'):
            query[column] = pd.to_numeric(query[column], errors='raise').astype(float)
        query['game_date'] = pd.to_datetime(query['game_date'], format='%Y-%m-%d')
        raw = self.profiles.raw91(query)
        x = self.ensemble.encode(raw)
        output = self.ensemble.predict(x, tokens, query[list(stream.KEY)].to_numpy(dtype=np.int64, copy=True))
        result.update(status='scored', probabilities=output['calibrated'][0].tolist(),
                      raw_mean=output['raw_mean'][0].tolist())
        return result
