"""Write the public publish contract's example files and JSON Schemas (apps/observer/publish_contract/v1/).

    PYTHONPATH=apps/observer/backend .venv-observer/bin/python apps/observer/scripts/export_publish_contract.py [target]

Everything comes from the synthetic world in ``backend/tests/publish_fixture.py``: the real publisher writes the
files and the schemas are inferred from those bodies. No network, no T7, no real game. ``test_publisher.py``
fails when the committed folder differs from what this script would write.
"""
from pathlib import Path
import sys
import tempfile

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP/'backend'))
sys.path.insert(0, str(APP/'backend/tests'))


def main(argv=None):
    import logging
    import publish_fixture
    logging.disable(logging.WARNING)  # the walk includes a failed ledger write and a dead feed on purpose
    argv = sys.argv[1:] if argv is None else argv
    target = Path(argv[0]) if argv else APP/'publish_contract/v1'
    with tempfile.TemporaryDirectory() as tmp:
        publish_fixture.export(target, tmp)
    print(target)


if __name__ == '__main__':
    main()
