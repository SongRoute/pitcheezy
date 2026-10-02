"""Scientific file enumeration for new policy/RL artifacts on macOS volumes.

Only AppleDouble sidecars with both the reserved filename prefix and binary
magic are metadata. A same-prefix ordinary payload remains a checked artifact.
No files are removed or changed, and existing manifests are never rewritten.
"""
from pathlib import Path


APPLEDOUBLE_MAGIC = bytes.fromhex('00051607')


def is_appledouble(path):
    path = Path(path)
    if not path.name.startswith('._') or not path.is_file():
        return False
    with path.open('rb') as stream:
        return stream.read(4) == APPLEDOUBLE_MAGIC


def artifact_names(directory, *, exclude=()):
    """Sorted relative file names, excluding only named paths and AppleDouble."""
    directory = Path(directory)
    excluded = set(exclude)
    return sorted(str(path.relative_to(directory)) for path in directory.rglob('*')
                  if path.is_file() and str(path.relative_to(directory)) not in excluded
                  and not is_appledouble(path))
