"""Korean-capable font for the figure scripts: Apple SD Gothic Neo -> Noto Sans CJK -> matplotlib default."""
from __future__ import annotations

import pathlib
import warnings

from matplotlib import font_manager as fm

CANDIDATES = [
    ('Apple SD Gothic Neo', ['/System/Library/Fonts/AppleSDGothicNeo.ttc']),
    ('Noto Sans CJK KR', ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                          '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
                          '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc']),
    ('Noto Sans KR', []),
    ('NanumGothic', []),
]


def korean_font() -> str | None:
    """Register and return the first available family name, or None (matplotlib default; Korean may show as boxes)."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for family, files in CANDIDATES:
        for f in files:
            if pathlib.Path(f).exists() and family not in installed:
                try:
                    fm.fontManager.addfont(f)
                    installed = {x.name for x in fm.fontManager.ttflist}
                except (OSError, RuntimeError):
                    pass
        if family in installed:
            return family
    warnings.warn('no Korean font found (tried Apple SD Gothic Neo, Noto Sans CJK); using the matplotlib default')
    return None


def rc(**extra):
    fam = korean_font()
    base = {'axes.unicode_minus': False}
    if fam:
        base['font.family'] = fam
    base.update(extra)
    return base
