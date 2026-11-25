"""Explicit video-adapter boundary; no video model is installed or inferred."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CVRequest:
    pitch_id: str
    media_ref: str | None = None
    setup_before_pitch: float | None = None


@dataclass(frozen=True)
class CVResult:
    status: str
    cv_status: str
    message: str
    source: str = 'none'


class CVAdapter(Protocol):
    def analyze(self, request: CVRequest) -> CVResult: ...


class NoMediaAdapter:
    def analyze(self, request: CVRequest) -> CVResult:
        if request.media_ref is not None:
            return CVResult('unavailable', 'unavailable:adapter_not_configured',
                            '영상 분석기가 연결되지 않았습니다. 영상 참조만으로 실제 목표를 추정하지 않습니다.')
        return CVResult('unavailable', 'unavailable:no_media',
                        '연결된 영상이 없어 자동 분석을 제공하지 않습니다. 직접 목표 구역을 입력할 수 있습니다.')
