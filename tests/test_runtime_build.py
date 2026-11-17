"""Runtime preparation rejects archives that cannot relocate before invoking Docker."""
import io
import os
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import unittest

from tests.suites import integration

ROOT = Path(__file__).resolve().parents[1]


@integration
class RuntimeBuildTest(unittest.TestCase):
    def test_archive_relocation_precondition_and_cleanup(self):
        for relocatable in (False, True):
            with self.subTest(relocatable=relocatable), TemporaryDirectory() as folder:
                root = Path(folder)
                archive, log = root/'environment.tar.gz', root/'build.log'
                with tarfile.open(archive, 'w:gz') as tar:
                    member = tarfile.TarInfo('bin/conda-unpack' if relocatable else 'bin/python')
                    member.size = 1
                    tar.addfile(member, io.BytesIO(b'\n'))
                docker = root/'docker'
                docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$BUILD_LOG"\n')
                docker.chmod(0o755)
                result = subprocess.run(['sh', str(ROOT/'scripts/build_paper_runtime.sh'),
                                         str(archive), 'test-runtime'], text=True, capture_output=True,
                    env={**os.environ, 'PATH': str(root)+os.pathsep+os.environ['PATH'], 'BUILD_LOG': str(log)})
                self.assertEqual(result.returncode == 0, relocatable, result.stderr)
                self.assertEqual(log.exists(), relocatable)
                if relocatable:
                    self.assertIn('test-runtime', log.read_text())
                else:
                    self.assertIn('relocatable conda-pack archive', result.stderr)
                self.assertFalse(list(root.glob('.bpc-paper-runtime.*')))
