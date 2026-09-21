"""Descriptive pre-pitch observations; no fatigue/condition diagnosis or model input."""
from __future__ import annotations

import math
import pandas as pd

from .domain import PITCH_LABELS

KEY = ['game_pk', 'at_bat_number', 'pitch_number']


class ObservedContext:
    def __init__(self, frame, lookback_days=90):
        columns = KEY+['game_date', 'pitcher', 'batter', 'pitch_type', 'release_speed']
        self.frame = frame[columns].copy()
        if self.frame.duplicated(KEY).any() or self.frame[KEY].isna().any().any():
            raise ValueError('Observed context requires unique, complete pitch identities')
        self.frame['game_date'] = pd.to_datetime(self.frame.game_date).dt.normalize()
        self.frame['release_speed'] = pd.to_numeric(self.frame.release_speed, errors='coerce')
        self.frame.loc[~self.frame.release_speed.map(lambda v: math.isfinite(v)), 'release_speed'] = float('nan')
        self.frame.sort_values(['game_date', *KEY], inplace=True)
        self.lookback_days = lookback_days
        self.games = {(int(game), int(pitcher)): part for (game, pitcher), part in
                      self.frame.groupby(['game_pk', 'pitcher'], sort=False)}
        self.pitchers = {int(pitcher): part for pitcher, part in self.frame.groupby('pitcher', sort=False)}
        self.baselines = {}

    def before(self, row, pitch_types):
        """Use only earlier recorded game pitches; output types do not depend on current actual type."""
        pitcher, game = int(row.pitcher), int(row.game_pk)
        date = pd.Timestamp(row.game_date).normalize()
        part = self.games[(game, pitcher)]
        past = part.loc[part.at_bat_number.lt(row.at_bat_number) |
                        (part.at_bat_number.eq(row.at_bat_number) & part.pitch_number.lt(row.pitch_number))]
        earlier_pas = past.loc[past.batter.eq(row.batter) & past.at_bat_number.lt(row.at_bat_number), 'at_bat_number']
        cache_key = (pitcher, date)
        if cache_key not in self.baselines:
            source = self.pitchers[pitcher]
            source = source.loc[source.game_date.lt(date) &
                                source.game_date.ge(date-pd.Timedelta(days=self.lookback_days))]
            self.baselines[cache_key] = source.groupby('pitch_type').release_speed.agg(['mean', 'count']).to_dict('index')
        baselines = self.baselines[cache_key]
        speeds = []
        for pitch_type in sorted(set(pitch_types)):
            # Select five PITCHES first, then omit missing speeds from their mean.
            # Otherwise missing observations silently lengthen the stated window.
            recent = past.loc[past.pitch_type.eq(pitch_type)].tail(5)
            measured = recent.release_speed.dropna()
            reference = baselines.get(pitch_type, {'mean': float('nan'), 'count': 0})
            recent_mean = float(measured.mean()) if len(measured) else None
            baseline_mean = float(reference['mean']) if reference['count'] else None
            speeds.append({'pitch_type': pitch_type, 'pitch_label': PITCH_LABELS.get(pitch_type, pitch_type),
                           'recent_pitch_count': len(recent), 'recent_measured_count': len(measured),
                           'recent_mean_mph': recent_mean, 'prior90_measured_count': int(reference['count']),
                           'prior90_mean_mph': baseline_mean,
                           'delta_mph': recent_mean-baseline_mean if recent_mean is not None and baseline_mean is not None else None})
        return {'prior_pitch_count': len(past), 'times_facing_batter': int(earlier_pas.nunique())+1,
                'speed_by_pitch_type': speeds,
                'speed_window': 'current_game_previous_up_to_5_same_type_pitches',
                'reference_window_days': self.lookback_days,
                'reference_cutoff': 'strictly_prior_date', 'condition_inference': False}


def context_notes(context):
    notes = [f"이 투구 전까지 던진 공 {context['prior_pitch_count']}개",
             f"이번 경기 이 타자와 {context['times_facing_batter']}번째 대결",
             '타자 성향·구종 구성은 경기 전날까지의 기록으로 계산']
    for speed in context['speed_by_pitch_type']:
        if speed['delta_mph'] is not None:
            notes.append(f"{speed['pitch_label']} 이전 최대 5구 중 구속 기록 {speed['recent_measured_count']}구 평균 "
                         f"{speed['recent_mean_mph']:.1f} mph · 이전 {context['reference_window_days']}일 평균 대비 "
                         f"{speed['delta_mph']:+.1f} mph")
    notes.append('구속 차이는 관측 기록이며 피로·부상·컨디션을 판정한 값이 아닙니다.')
    return notes
