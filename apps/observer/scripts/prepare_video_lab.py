"""Run a local approved, manual-seed pre-release video baseline into a new immutable result."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'apps/observer/backend'))

from observer_app.video_lab import LAB_ROOT, resolve_annotation_clip, validate_annotation, track_video


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='New result directory under v2/video-lab; default timestamp/version')
    parser.add_argument('--search-radius', type=int, default=40)
    args = parser.parse_args()
    if not Path('/Volumes/T7 Shield').is_mount():
        raise SystemExit('Mounted approved SSD required')
    annotation_path = args.annotation.resolve()
    annotation = validate_annotation(json.loads(annotation_path.read_text()))
    clip = resolve_annotation_clip(annotation)
    clip_hash = sha256(clip)
    if clip_hash != annotation['clip_sha256']:
        raise ValueError('Clip hash differs from the annotated media identity')
    result_root = LAB_ROOT/'video-lab'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = (args.output or result_root/f"annotation-v{annotation['annotation_version']}-{stamp}").resolve()
    if not output.is_relative_to(result_root.resolve()) or output == result_root.resolve():
        raise ValueError('Output must be a new subdirectory inside approved v2/video-lab')
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), REPO/'apps/observer/backend/observer_app/video_lab.py']
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in sources}
    annotation_hash = sha256(annotation_path)
    config = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'annotation_file': str(annotation_path),
              'annotation_sha256': annotation_hash, 'clip_path': str(clip), 'clip_sha256': clip_hash,
              'source_url': annotation['source_url'], 'source_hashes': hashes,
              'search_radius_pixels': args.search_radius, 'network_access': False,
              'purpose': 'Manual-seed image tracking diagnostic; no independent ground truth or physical plate coordinates'}
    (output/'annotation.json').write_text(json.dumps(annotation, ensure_ascii=False, indent=2, allow_nan=False))
    (output/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    result = track_video(clip, annotation, search_radius=args.search_radius)
    identities_unchanged = (sha256(clip) == clip_hash and sha256(annotation_path) == annotation_hash and
                            hashes == {str(path.relative_to(REPO)): sha256(path) for path in sources})
    if not identities_unchanged:
        raise RuntimeError('Media, annotation, or source changed during tracking')
    result['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    result['integrity_ok'] = True
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'output': str(output), 'status': result['status'], 'abstain_reason': result['abstain_reason'],
                      'processed_frames': result['processed_frames'], 'matched_frames': result['matched_frames']}), flush=True)


if __name__ == '__main__':
    main()
