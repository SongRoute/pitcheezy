"""Public representations are constructed explicitly; raw future rows stay private."""
import math

PITCH_LABELS = {'FF': '포심', 'SI': '싱커', 'FC': '커터', 'SL': '슬라이더', 'ST': '스위퍼',
                'SV': '슬러브', 'CU': '커브', 'KC': '너클커브', 'CH': '체인지업', 'FS': '스플리터'}
RESULT_LABELS = {'ball': '볼', 'blocked_ball': '볼', 'called_strike': '루킹 스트라이크',
    'swinging_strike': '헛스윙', 'swinging_strike_blocked': '헛스윙', 'foul': '파울', 'foul_tip': '파울팁',
    'hit_into_play': '인플레이', 'strikeout': '삼진', 'walk': '볼넷', 'hit_by_pitch': '몸에 맞는 공',
    'single': '안타', 'double': '2루타', 'triple': '3루타', 'home_run': '홈런', 'field_out': '타구 아웃',
    'force_out': '포스 아웃', 'fielders_choice_out': '야수 선택 아웃', 'sac_fly': '희생 플라이',
    'grounded_into_double_play': '병살타', 'double_play': '병살', 'sac_fly_double_play': '희생플라이 병살'}
ZONES = [{'id': f'{height}_{side}', 'label': f'{h_label} {s_label}', 'column': column, 'row': row}
         for row, (height, h_label) in enumerate([('low', '낮은'), ('middle', '가운데 높이'), ('high', '높은')])
         for column, (side, s_label) in enumerate([('left', '왼쪽'), ('middle', '중앙'), ('right', '오른쪽')])]
ZONE_BY_ID = {zone['id']: zone for zone in ZONES}


def target_point(zone_id, bounds):
    zone = ZONE_BY_ID[zone_id]
    return {'x': (zone['column']-1)*1.66/3,
            'z': bounds['bottom']+(zone['row']+.5)*(bounds['top']-bounds['bottom'])/3}


def actual_zone(x, z, bounds):
    if x is None or z is None or not math.isfinite(x) or not math.isfinite(z):
        return '위치 정보 없음'
    if abs(x) > .83 or z < bounds['bottom'] or z > bounds['top']:
        return '존 바깥'
    column = min(2, max(0, int((x+.83)/1.66*3)))
    row = min(2, max(0, int((z-bounds['bottom'])/(bounds['top']-bounds['bottom'])*3)))
    return ZONES[row*3+column]['label']


def spatial_comparison(pitch, zone_id):
    bounds = pitch['zone_bounds']
    target = target_point(zone_id, bounds)
    rec = pitch.get('recommendation') or {}
    candidates = rec.get('candidates') or []
    recommended = candidates[0]['zone_label'] if candidates else '추천 정보 없음'
    x, z = pitch.get('x'), pitch.get('z')
    distance = None if x is None or z is None else round(math.hypot(
        (x-target['x'])/1.66, (z-target['z'])/(bounds['top']-bounds['bottom'])), 3)
    return {'recommended_zone_label': recommended, 'intended_zone_label': ZONE_BY_ID[zone_id]['label'],
            'actual_zone_label': actual_zone(x, z, bounds), 'target_error_zone_units': distance,
            'interpretation': '사용자가 입력한 목표 구역과 실제 도달 위치의 공간적 비교입니다. 선수의 인과적 책임 점수는 아닙니다.'}
