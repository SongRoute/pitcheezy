#!/usr/bin/env python3
"""Check selected Markdown and derive a navigation graph; stdlib only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def git_files(root: Path, base: str | None = None) -> list[str]:
    if base:
        if base.startswith('-'):
            raise ValueError('base must be a Git commit reference, not an option')
        base = subprocess.check_output(['git', 'rev-parse', '--verify', '--end-of-options', base + '^{commit}'], cwd=root, stderr=subprocess.PIPE).decode().strip()
    command = ['git', 'diff', '--name-only', '--diff-filter=ACMR', '-z', base, '--'] if base else ['git', 'ls-files', '-z']
    return [p for p in subprocess.check_output(command, cwd=root).decode().split('\0') if p.endswith('.md')]


def links(source: str):
    """Inline and full/collapsed reference links outside fenced/inline code.

    Deliberately bounded Markdown subset: HTML, shortcut references and anchors
    are not validated. This is a path check, not a general Markdown renderer.
    """
    lines = []
    fence = None
    for line in source.splitlines():
        line = re.sub(r'^(?:\s{0,3}> ?)+', '', line)
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if marker and fence is None:
            fence = marker.group(1)
            lines.append('')
            continue
        if fence is not None:
            if re.match(r'^\s{0,3}' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}\s*$', line):
                fence = None
            lines.append('')
            continue
        lines.append(re.sub(r'(`+).*?\1', '', line))
    text = '\n'.join(lines)
    definitions = {}
    for match in re.finditer(r'^\s{0,3}\[([^\]]+)\]:\s*(?:<([^>]+)>|(\S+))', text, re.M):
        definitions.setdefault(' '.join(match[1].lower().split()), match[2] or match[3])
    # Parentheses in destinations must be angle-wrapped or escaped.
    for match in re.finditer(r'\[([^\]\n]+)\]\(\s*(?:<([^>\n]+)>|((?:\\.|[^\s()])+))(?:\s+["\'][^\n]*?["\'])?\s*\)', text):
        if (len(text[:match.start()]) - len(text[:match.start()].rstrip('\\'))) % 2:
            continue
        yield match[1], re.sub(r'\\([() ])', r'\1', match[2] or match[3])
    for match in re.finditer(r'\[([^\]\n]+)\]\[([^\]\n]*)\]', text):
        if (len(text[:match.start()]) - len(text[:match.start()].rstrip('\\'))) % 2:
            continue
        key = ' '.join((match[2] or match[1]).lower().split())
        yield match[1], definitions.get(key)  # None is an undefined reference.


def target(root: Path, source: str, url: str):
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path or parts.path.startswith('/'):
        return None  # External, fragment-only and host-absolute paths are outside this check.
    path = (root / source).parent / unquote(parts.path)
    normalized = path.resolve()
    try:
        relative = normalized.relative_to(root.resolve())
    except ValueError:
        return None  # Existing worktree/external-data references are not repo navigation.
    return relative.as_posix(), path.exists()


def check(root: Path, paths: list[str], exceptions: dict[str, str]):
    errors = []
    checked_links = 0
    lengths = {}
    for name in sorted(set(paths)):
        path = root / name
        if not path.is_file():
            errors.append(f'{name}: missing document')
            continue
        text = path.read_text(encoding='utf-8')
        lengths[name] = len(text)
        if len(text) > 3000 and not exceptions.get(name, '').strip():
            errors.append(f'{name}: {len(text)} characters exceeds 3000; split or record a preservation reason')
        for label, url in links(text):
            if url is None:
                errors.append(f'{name}: undefined reference [{label}]')
                continue
            try:
                resolved = target(root, name, url)
            except ValueError:
                errors.append(f'{name}: malformed destination {url}')
                continue
            if resolved:
                checked_links += 1
                if not resolved[1]:
                    errors.append(f'{name}: missing target {url}')
    if (root / 'CLAUDE.md').read_bytes() != b'@AGENTS.md\n':
        errors.append('CLAUDE.md: expected exactly @AGENTS.md followed by newline')
    return {'documents': len(lengths), 'characters': lengths, 'local_links': checked_links, 'errors': errors}


def graph(root: Path):
    nodes = []
    edges = []
    errors = []
    for name in sorted(git_files(root)):
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8')
        nodes.append({'path': name, 'characters': len(text)})
        for label, url in links(text):
            if url is None:
                continue
            try:
                resolved = target(root, name, url)
            except ValueError:
                errors.append({'source': name, 'destination': url, 'error': 'malformed destination'})
                continue
            if resolved:
                edges.append({'from': name, 'to': resolved[0], 'label': label, 'exists': resolved[1]})
    return {'schema': 'pitcheezy-doc-links-v1', 'source': 'tracked Markdown links; navigation only',
            'limitations': ['no HTML/shortcut-reference/anchor/HTTP checks', 'host-absolute/outside-repo paths excluded'],
            'nodes': nodes, 'edges': edges, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', help='repository-relative Markdown paths')
    parser.add_argument('--base', help='check changed tracked Markdown against this Git ref')
    parser.add_argument('--graph', type=Path, help='write derived graph to a new local JSON file')
    args = parser.parse_args()
    if not args.paths and not args.base and not args.graph:
        parser.error('give paths, --base, or --graph')
    exceptions_path = ROOT / 'docs/maintenance/length-exceptions.json'
    exceptions = json.loads(exceptions_path.read_text()) if exceptions_path.exists() else {}
    if not isinstance(exceptions, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in exceptions.items()):
        parser.error('length exceptions must map exact paths to nonempty reasons')
    try:
        paths = args.paths + (git_files(ROOT, args.base) if args.base else [])
    except (ValueError, subprocess.CalledProcessError):
        parser.error('--base must name an existing Git commit')
    if any(not p.endswith('.md') or Path(p).is_absolute() or '..' in Path(p).parts for p in paths):
        parser.error('document paths must be repository-relative .md files without ..')
    result = check(ROOT, paths, exceptions)
    if args.graph:
        try:
            data = graph(ROOT)  # Finish reading before creating an output artifact.
            args.graph.parent.mkdir(parents=True, exist_ok=True)
            # Do not overwrite a source or a prior evidence artifact.
            with args.graph.open('x', encoding='utf-8') as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
        except (OSError, UnicodeError, subprocess.CalledProcessError) as exc:
            parser.error(f'cannot produce graph: {exc}')
        result['graph'] = str(args.graph)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(result['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
