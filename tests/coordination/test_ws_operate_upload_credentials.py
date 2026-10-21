"""Offline launcher-to-transport credential regressions; no publisher or HTTP request.

Run directly with Python -I -B; --source-root can select another reviewed checkout.
The file-only case is intentionally red on 000e4ba0 until the launcher is fixed.
"""
import argparse
from contextlib import ExitStack
import http.client  # Load SSL before socket guards are installed.
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.request


SOURCE_ROOT = Path(__file__).resolve().parents[2]


class ForbiddenIO(BaseException):
    """A boundary violation cannot be mistaken for a caught configuration error."""


class UploadCredentials(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.blocked = []

        def forbid(*args, **kwargs):
            self.blocked.append("network/process attempt")
            raise ForbiddenIO("real network/process execution forbidden")

        for obj, names in (
            (socket, ("socket", "create_connection", "getaddrinfo")),
            (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
            (os, ("system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp")),
        ):
            for name in names:
                if hasattr(obj, name):
                    self.stack.enter_context(patch.object(obj, name, forbid))
        self.stack.enter_context(patch.object(os, "environ", {}))
        self.ws = self.load("ops/ws_operate.py", "ws_operate")
        self.publisher = self.load("backend/observer_app/publisher.py", "publisher")
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="cx45-")))
        self.token = self.tmp / "synthetic-token"
        self.token.write_text("F" * 45)
        self.token.chmod(0o600)
        self.cfg = {
            "code_dir": str(SOURCE_ROOT),
            "publish_config": "apps/observer/publish_config.json",
            "fixed_delay_s": 15,
            "upload": {"allowed": True},
        }
        os.environ[self.ws.BASE_URL_ENV] = "https://example.invalid/_publish"

    @staticmethod
    def load(relative, name):
        path = SOURCE_ROOT / "apps/observer" / relative
        module = types.ModuleType(name)
        module.__file__ = str(path)
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        return module

    def tearDown(self):
        self.assertEqual(self.blocked, [], "a forbidden boundary was reached")

    def check_credentials(self, *, env_token=False, file_token=False):
        if env_token:
            os.environ[self.ws.TOKEN_ENV] = "E" * 45
        if file_token:
            os.environ[self.ws.TOKEN_FILE_ENV] = str(self.token)
        enabled, problems, _ = self.ws.upload_decision(self.cfg, True)
        self.assertTrue(enabled, problems)
        self.assertEqual(problems, [])
        config = self.ws.runtime_publish_config(self.cfg, enabled)
        remote = self.publisher.HttpSink(config["upload"])
        self.assertEqual(remote._authorization, "Bearer " + ("E" if env_token else "F") * 45)
        self.assertEqual(config["upload"]["token_env"], self.ws.TOKEN_ENV if env_token else None)
        self.assertEqual(config["upload"]["token_file"], None if env_token else str(self.token.resolve()))

    def test_environment_token(self):
        self.check_credentials(env_token=True)

    def test_file_token(self):
        self.check_credentials(file_token=True)

    def test_environment_token_takes_precedence_over_file(self):
        self.check_credentials(env_token=True, file_token=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args, rest = parser.parse_known_args()
    SOURCE_ROOT = args.source_root.resolve()
    unittest.main(argv=[sys.argv[0], *rest])
