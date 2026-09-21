"""Materialize a hash-verified local scientific runtime without research CLI imports.

Definitions selected below retain their original source text. The generated
modules preserve pickle module/class names. This prepares code, not model weights,
and makes no claim of cross-platform portability before isolated regression.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys

REPO = Path(__file__).resolve().parents[3]
APP = REPO/'apps/observer'
DEFAULT_BUNDLE = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/minimal-pitch-service-v1')

COPIES = ['pitchmdp/__init__.py', 'pitchmdp/planner.py', 'pitchmdp/game.py',
          'pitchmdp/sequence_model.py', 'pitchmdp/sequence_delivery.py', 'pitchmdp/archetypes.py']

# Target -> source path, names in source order, replacement import prelude.
EXTRACTIONS = {
    'pitchmdp/sequence_data.py': ('pitchmdp/sequence_data.py',
        ['PHYSICAL_COLUMNS', 'PHYSICAL_CHANNELS', '_physical_values', 'PhysicalNormalizer'],
        'import numpy as np\nimport pandas as pd\n'),
    'pitchmdp/model.py': ('pitchmdp/model.py', ['OUTCOMES', 'outcome_labels', 'CountBaseline'],
        'import numpy as np\nimport pandas as pd\n'),
    'representation_adapters.py': ('scripts/representation_adapters.py',
        ['_require_train', '_base_features', '_batter_keys', 'BatterContext'],
        'import numpy as np\nimport pandas as pd\nfrom pitchmdp.archetypes import Archetypes\n'),
    'run_sequence_frequency_baselines.py': ('scripts/run_sequence_frequency_baselines.py',
        ['COUNT_KEYS', 'TYPE_KEYS', 'PITCHER_KEYS', 'HierarchicalFrequencyBaseline', 'temperature_predictions'],
        'import numpy as np\nimport pandas as pd\nfrom scipy.special import softmax\n'
        'from pitchmdp.model import CountBaseline, OUTCOMES, outcome_labels\n'),
    'run_sequence_context_frequency.py': ('scripts/run_sequence_context_frequency.py',
        ['CONTEXT_KEYS', 'PITCHER_CONTEXT_KEYS', 'ContextFrequencyBaseline'],
        'import numpy as np\nimport pandas as pd\nfrom pitchmdp.model import outcome_labels\n'
        'from run_sequence_frequency_baselines import TYPE_KEYS, HierarchicalFrequencyBaseline\n'),
    'diagnose_sequence_legality.py': ('scripts/diagnose_sequence_legality.py', ['condition_on_legality'],
        'import numpy as np\n'),
    'minimal_pitch_service.py': ('scripts/minimal_pitch_service.py',
        ['ASSUMPTIONS', 'RequestError', '_integer', '_iso_date', 'validate_request', 'evaluate_policy', 'Engine'],
        'from datetime import date\nimport hashlib\nimport json\nfrom pathlib import Path\n'
        'import pickle\nimport threading\nimport time\nimport numpy as np\nimport pandas as pd\n'
        'from scipy.special import softmax\nimport torch\n'
        'from pitchmdp.archetypes import STYLE_COLUMNS, RELIABILITY_COLUMNS, STYLE_NAMES\n'
        'from pitchmdp.game import GameState, terminal_values\nfrom pitchmdp.planner import OUTCOMES, solve_pa\n'
        'from pitchmdp.sequence_model import SequenceModel, SequenceNetwork\n'
        'from diagnose_sequence_legality import condition_on_legality\n'
        'from run_sequence_frequency_baselines import temperature_predictions\n'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_definitions(source_text, names):
    """Exact AST-bounded text, not ast.unparse or a behavioral rewrite."""
    source_lines = source_text.splitlines(keepends=True)
    found = {}
    for node in ast.parse(source_text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            identifiers = [target.id for target in targets if isinstance(target, ast.Name)]
        else:
            continue
        wanted = set(identifiers).intersection(names)
        if wanted:
            first = min([node.lineno, *[decorator.lineno for decorator in getattr(node, 'decorator_list', [])]])
            segment = ''.join(source_lines[first-1:node.end_lineno])
            for name in wanted:
                if name in found:
                    raise ValueError('Duplicate selected source symbol: '+name)
                found[name] = {'source': segment, 'first_line': first, 'last_line': node.end_lineno,
                               'sha256': hashlib.sha256(segment.encode()).hexdigest()}
    missing = set(names).difference(found)
    if missing:
        raise ValueError('Missing captured symbols: '+','.join(sorted(missing)))
    return found


def dependency_snapshot(destination):
    """Pin installed active dependency closure and copy only supplied license files."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    # Pickled pandas tables retain PyArrow-backed dtype/index metadata even
    # though the prediction module import graph never reads parquet data.
    queue = ['numpy', 'pandas', 'scipy', 'torch', 'pyarrow']
    resolved = {}
    while queue:
        name = canonicalize_name(queue.pop(0))
        if name in resolved:
            continue
        distribution = metadata.distribution(name)
        active = []
        for requirement in distribution.requires or []:
            parsed = Requirement(requirement)
            if parsed.marker is None or parsed.marker.evaluate({'extra': ''}):
                active.append(parsed.name)
                queue.append(parsed.name)
        notices = []
        for relative in distribution.files or []:
            basename = Path(str(relative)).name.upper()
            # Include nested licenses furnished by the installed wheel (BLAS, etc.).
            if not basename.startswith(('LICENSE', 'LICENCE', 'NOTICE', 'COPYING')):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            safe_relative = Path(*[part for part in Path(str(relative)).parts if part not in ('.', '..', '/')])
            target = destination/'third_party_notices'/name/safe_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            notices.append({'path': str(target.relative_to(destination)), 'sha256': sha256(target),
                            'installed_distribution_path': str(relative)})
        resolved[name] = {'version': distribution.version, 'requires_active': sorted(active),
                          'license_metadata': distribution.metadata.get('License-Expression') or distribution.metadata.get('License'),
                          'license_files': notices,
                          'notice_status': 'copied_supplied_files' if notices else 'no_supplied_license_file_found'}
    return resolved


def prepare_runtime(bundle, output, *, include_dependency_snapshot=True):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    capture = bundle/'source'
    pinned = json.loads((bundle/'source_hashes.json').read_text())
    for relative, expected in pinned.items():
        path = (capture/relative).resolve()
        if not path.is_relative_to(capture) or sha256(path) != expected:
            raise ValueError('Frozen captured source mismatch: '+relative)
    output.mkdir(parents=True, exist_ok=False)
    artifacts, extraction_manifest = {}, {}
    for relative in COPIES:
        if relative not in pinned:
            raise ValueError('Source not pinned in original bundle: '+relative)
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(capture/relative, target)
        artifacts[relative] = sha256(target)
        extraction_manifest[relative] = {'mode': 'byte_identical_copy', 'source': relative, 'source_sha256': pinned[relative]}
    for relative, (source, names, imports) in EXTRACTIONS.items():
        if source not in pinned:
            raise ValueError('Source not pinned in original bundle: '+source)
        text = (capture/source).read_text()
        selected = extract_definitions(text, names)
        banner = ('"""Generated prediction runtime: exact selected definitions from a hash-verified capture.\n'
                  'Training methods may remain for class identity; the service does not invoke them.\n"""\n'
                  'from __future__ import annotations\n\n')
        body = banner+imports+'\n\n'+'\n\n'.join(selected[name]['source'].rstrip('\n') for name in names)+'\n'
        # Parse generated source before persistence without importing research modules.
        ast.parse(body)
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        artifacts[relative] = sha256(target)
        extraction_manifest[relative] = {'mode': 'exact_definition_extraction_with_explicit_import_prelude',
            'source': source, 'source_sha256': pinned[source], 'import_prelude': imports,
            'symbols': {name: {key: value for key, value in info.items() if key != 'source'} for name, info in selected.items()}}
    dependencies = dependency_snapshot(output) if include_dependency_snapshot else {}
    if dependencies:
        (output/'requirements-scientific.lock.txt').write_text(''.join(f'{name}=={info["version"]}\n' for name, info in sorted(dependencies.items())))
    # No license is invented for project-owned research code.
    project_notices = []
    for pattern in ('LICENSE*', 'NOTICE*', 'COPYING*'):
        for source in REPO.glob(pattern):
            if source.is_file():
                target = output/'project_notices'/source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                project_notices.append({'path': str(target.relative_to(output)), 'sha256': sha256(target)})
    manifest = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'bundle_manifest_sha256': sha256(bundle/'bundle_manifest.json'),
        'source_manifest_sha256': sha256(bundle/'source_hashes.json'), 'source_hashes': pinned,
        'runtime_files': artifacts, 'extraction': extraction_manifest,
        'preparer_sha256': sha256(Path(__file__)), 'dependencies': dependencies,
        'environment_at_preparation': {'python': platform.python_version(), 'platform': platform.platform(),
                                       'machine': platform.machine()},
        'project_notices': project_notices, 'project_license_status': 'copied_existing_notice' if project_notices else 'no_project_license_file_found',
        'scope': 'Local prediction runtime; no original experiments imports or source modifications; validation pending',
        'training_invoked': False, 'portable_verified': False}
    (output/'runtime_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--output', type=Path, default=APP/'runtime_src')
    args = parser.parse_args()
    output = args.output.resolve()
    if output != (APP/'runtime_src').resolve():
        raise SystemExit('CLI output is the new apps/observer/runtime_src directory only')
    manifest = prepare_runtime(args.bundle, output)
    print(json.dumps({'runtime': str(output), 'modules': len(manifest['runtime_files']),
                      'pinned_dependencies': len(manifest['dependencies']), 'validation': 'pending'}), flush=True)


if __name__ == '__main__':
    main()
