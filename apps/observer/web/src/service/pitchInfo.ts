/** One-line pitch-type explanations and the 2026 MLB regular-season average speed per type (final-presentation proposal 2026-11-19 §2–§4).
 *  Speeds are MEASURED, not estimated: Statcast 2026 frozen parquet (holdout_2026/statcast_2026.parquet, 716,792 pitches, all game_type R;
 *  2,655 rows without release_speed left out), mean release_speed per pitch_type: avgMph = that mean (1 decimal), avgKmh = mean × 1.609344
 *  to whole km/h, half up (PM recompute 2026-11-19, pyarrow dropna: FF 152.52, SI 151.50 → 152, FC 144.54, SL 138.95, CH 138.46 → 138,
 *  FS 139.42, ST 133.27, KC 132.94, SV 132.32, CU 129.20).
 *  Types without a measured value here (CS, FO, KN, …) show no line rather than a guess. */
export type PitchInfo = { desc: string; avgKmh: number; avgMph: number };
export const PITCH_INFO: Record<string, PitchInfo> = {
  FF: { desc: '가장 빠른 직구. 덜 떨어지며 들어와요.', avgKmh: 153, avgMph: 94.8 },
  SI: { desc: '직구처럼 오다 가라앉으며 휘어요.', avgKmh: 152, avgMph: 94.1 },
  FC: { desc: '직구 속도로 오다 끝에서 살짝 꺾여요.', avgKmh: 145, avgMph: 89.8 },
  SL: { desc: '옆으로 꺾이는 빠른 변화구.', avgKmh: 139, avgMph: 86.3 },
  ST: { desc: '옆으로 크게 휘는 슬라이더.', avgKmh: 133, avgMph: 82.8 },
  CU: { desc: '위에서 아래로 크게 떨어지는 느린 공.', avgKmh: 129, avgMph: 80.3 },
  KC: { desc: '더 세게 떨어지는 커브.', avgKmh: 133, avgMph: 82.6 },
  CH: { desc: '직구와 같은 팔 동작, 속도만 뚝 줄인 공.', avgKmh: 138, avgMph: 86.0 },
  FS: { desc: '직구처럼 오다 앞에서 뚝 떨어져요.', avgKmh: 139, avgMph: 86.6 },
  SV: { desc: '슬라이더와 커브의 중간.', avgKmh: 132, avgMph: 82.2 },
};
export const pitchInfo = (code: string | null | undefined): PitchInfo | null => (code && PITCH_INFO[code.toUpperCase()]) || null;
export const avgSpeedText = (info: PitchInfo, unit: 'kmh' | 'mph' = 'kmh') => unit === 'mph' ? `${info.avgMph.toFixed(1)}mph` : `${info.avgKmh}km/h`;
export const pitchInfoLine = (info: PitchInfo, unit: 'kmh' | 'mph' = 'kmh') => `${info.desc} 평균 ${avgSpeedText(info, unit)}`;
export const PITCH_AVG_NOTE = '평균은 2026 MLB 정규시즌 전체 투구 기준이에요.';

