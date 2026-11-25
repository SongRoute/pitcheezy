"""Predictor factory for observer_app.prepitch_prediction.PredictionWorker (runs inside the worker process only).

    PredictionWorker('pitcheezy.svc_p9.worker_predictor:make')
    worker.submit(pitch_key, {'feed_bytes': raw_body, 'source_sha256': sha256(raw_body), 'game_pk': pk, 'game_date': 'YYYY-MM-DD'})

The snapshot travels as the ORIGINAL body bytes plus their sha256 (code review P1-4), never as a live dict; the bytes
are checked against the sha and parsed here (gzip or plain JSON).

``make`` activates the vendored research snapshot FIRST (the worker is a fresh spawn process), loads the pin-verified
B_PSR9N6 bundle (W2) and the september state encoder, and returns ``predict(pitch_key, inputs)``. ``predict`` runs the
W1 converter and answers W1's OWN pitch key; PredictionWorker compares it with the submitted key and keeps no card
when they differ (code review P1-3). An omission raises ``OmitPrediction`` so the worker records no card.
"""
from __future__ import annotations


import gzip
import hashlib
import json


class OmitPrediction(Exception):
    """W1 could not build inputs for this pitch: no card (the reason is the message)."""


def make(threads=2):
    from . import predictor as P
    P.activate_snapshot()                      # before anything imports ``experiments``
    bundle = P.Bundle.load(threads=threads)
    from . import live_inputs as L            # imports seq_shadow -> experiments.* from the snapshot
    P.check_loaded_modules()
    encoder = L.Encoder11.load()
    converters = {}
    counts = {}

    def predict(pitch_key, inputs):
        pk, date = int(inputs['game_pk']), str(inputs['game_date'])
        raw, sha = inputs.get('feed_bytes'), inputs.get('source_sha256')
        if not isinstance(raw, (bytes, bytearray)) or hashlib.sha256(raw).hexdigest() != sha:
            counts['payload_sha_mismatch'] = counts.get('payload_sha_mismatch', 0) + 1
            raise OmitPrediction('payload_sha_mismatch')
        feed = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
        conv = converters.get(pk)
        if conv is None:
            conv = converters[pk] = L.LiveInputs(pk, date, encoder)
        x = conv.convert(feed, sha)
        if isinstance(x, L.Omit):
            counts[x.reason] = counts.get(x.reason, 0) + 1
            raise OmitPrediction(x.reason)
        out = bundle.predict(*x.batch())
        return {'pitch_key': x.pitch_key, 'probabilities': out['q'][0].tolist(), 'fallback': bool(out['fallback'][0])}

    predict.omissions = counts
    return predict
