"""Exercise Splash's user-local release lifecycle without building or loading models."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import splash_runtime


def digest(path: Path) -> str:
    """Return a SHA-256 digest for a fixture file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalInstallTests(unittest.TestCase):
    """Test the fork installer using private temporary package and home trees."""

    def setUp(self) -> None:
        """Create isolated fixture roots for one lifecycle test."""
        self.temporary = tempfile.TemporaryDirectory(
            prefix="splash-local-install-test-"
        )
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.paths = splash_runtime.RuntimePaths.defaults(self.home)

    def tearDown(self) -> None:
        """Remove only this test's temporary fixture tree."""
        self.temporary.cleanup()

    def _qualification(self, source: Path, revision: str) -> Path:
        """Write local-gate and private-package fixture evidence for one revision."""
        attempt = (self.root / "evidence" / revision).resolve()
        attempt.mkdir(parents=True)
        temporary_root = self.root / f"splash-qualification-fixture-{revision[:8]}"
        gates = []
        for index, target in enumerate(splash_runtime.QUALIFICATION_TARGETS, start=1):
            stdout_path = attempt / f"{index}.stdout"
            stderr_path = attempt / f"{index}.stderr"
            stdout_path.write_text(f"{target} stdout\n", encoding="utf-8")
            stderr_path.write_text(f"{target} stderr\n", encoding="utf-8")
            gates.append(
                {
                    "target": target,
                    "argv": splash_runtime._expected_gate_argv(target, temporary_root),
                    "cwd": str(source.resolve()),
                    "return_code": 0,
                    "heavy_op_need_gb": splash_runtime.HEAVY_OP_NEED_GB,
                    "temporary_root": str(temporary_root),
                    "status": "pass",
                    "stdout": {
                        "path": str(stdout_path),
                        "size": stdout_path.stat().st_size,
                        "sha256": digest(stdout_path),
                    },
                    "stderr": {
                        "path": str(stderr_path),
                        "size": stderr_path.stat().st_size,
                        "sha256": digest(stderr_path),
                    },
                }
            )
        revision_archive = (
            source / "dist" / f"splash-local-{revision}-arm64-macos26.tar.gz"
        )
        evidence_archive = attempt / revision_archive.name
        shutil.copyfile(revision_archive, evidence_archive)
        package_source = (
            self.root / f"splash-qualified-package-fixture-{revision}" / "source"
        )
        package_stdout = attempt / "07-package.stdout"
        package_stderr = attempt / "07-package.stderr"
        package_stdout.write_text("package stdout\n", encoding="utf-8")
        package_stderr.write_text("package stderr\n", encoding="utf-8")
        package_argv = [
            str(Path.home() / ".local/bin/heavy-op"),
            "make",
            "package",
            f"RELEASE_VERSION=local-{revision}",
            "PYTHON_CANDIDATES=python3",
        ]
        package_streams = {
            "stdout": {
                "path": str(package_stdout),
                "size": package_stdout.stat().st_size,
                "sha256": digest(package_stdout),
            },
            "stderr": {
                "path": str(package_stderr),
                "size": package_stderr.stat().st_size,
                "sha256": digest(package_stderr),
            },
        }
        receipt = {
            "schema": splash_runtime.QUALIFICATION_SCHEMA,
            "status": "pass",
            "source_revision": revision,
            "source_remote": "origin",
            "source_ref": "refs/remotes/origin/main",
            "qualified_at": "2026-10-12T12:00:00Z",
            "qualification_source": "local-ci-equivalent",
            "source_path": str(source.resolve()),
            "origin_url": "https://github.com/jecruz/splash.git",
            "source_recheck": {"status": "pass", "revision": revision},
            "hosted_ci": {"provider": "woodpecker", "status": "not-configured"},
            "heavy_op": {"need_gb": splash_runtime.HEAVY_OP_NEED_GB},
            "required_targets": list(splash_runtime.QUALIFICATION_TARGETS),
            "gates": gates,
            "package_artifact": {
                "status": "pass",
                "source_revision": revision,
                "source_path": str(package_source),
                "source_recheck_before": {"status": "pass", "revision": revision},
                "source_recheck_after": {"status": "pass", "revision": revision},
                "argv": package_argv,
                "cwd": str(package_source),
                "return_code": 0,
                "heavy_op_need_gb": splash_runtime.HEAVY_OP_NEED_GB,
                "archive": {
                    "path": str(evidence_archive),
                    "size": evidence_archive.stat().st_size,
                    "sha256": digest(evidence_archive),
                },
                "worktree_cleanup": {"status": "pass"},
                "temporary_cleanup": {"status": "pass"},
                "command_evidence": {
                    "argv": package_argv,
                    "cwd": str(package_source),
                    "return_code": 0,
                    "heavy_op_need_gb": splash_runtime.HEAVY_OP_NEED_GB,
                    **package_streams,
                },
            },
        }
        receipt_path = attempt / "qualification.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        return receipt_path

    def _package(self, source: Path, revision: str) -> Path:
        """Create a deterministic fake runtime archive for a source revision."""
        version = splash_runtime._source_version(source, revision)
        name = f"splash-{version}-arm64-macos26"
        package_root = self.root / f"package-{revision}"
        (package_root / "engine").mkdir(parents=True)
        (package_root / "python/bin").mkdir(parents=True)
        (package_root / "install").mkdir(parents=True)
        binary = package_root / "engine/splash"
        metallib = package_root / "engine/splash.metallib"
        binary.write_bytes(b"fixture engine binary\n")
        metallib.write_bytes(b"fixture metallib\n")
        (package_root / "release.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "binary_sha256": digest(binary),
                    "metallib_sha256": digest(metallib),
                }
            ),
            encoding="utf-8",
        )
        python = package_root / "python/bin/python3"
        python.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            "  *platform.python_version*) echo 3.13.15 ;;\n"
            f"  *) echo 'Splash {version}' ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        (package_root / "install/launcher.py").write_text(
            "# fixture\n", encoding="utf-8"
        )
        stable = Path(__file__).resolve().parents[3] / "install/splash-engine"
        shutil.copy2(stable, package_root / "install/splash-engine")
        (package_root / "install/models.py").write_text("# fixture\n", encoding="utf-8")
        (package_root / "install/splash-engine").chmod(0o755)

        dist = source / "dist"
        dist.mkdir(parents=True, exist_ok=True)
        archive_path = dist / f"{name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(package_root, arcname=name)
        archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
            digest(archive_path) + "\n", encoding="ascii"
        )
        return archive_path

    def test_install_update_verify_rollback_uninstall_and_legacy_preservation(self):
        """Exercise atomic current/previous transitions and exact ownership cleanup."""
        revisions = (
            "1" * 40,
            "2" * 40,
        )
        sources = []
        qualifications = []
        for revision in revisions:
            source = self.root / revision
            source.mkdir()
            self._package(source, revision)
            sources.append(source)
            qualifications.append(self._qualification(source, revision))

        legacy_shim = self.paths.launcher.with_name("splash-1974")
        legacy_shim.parent.mkdir(parents=True)
        legacy_shim.write_text("legacy shim\n", encoding="utf-8")
        legacy_release = self.home / ".local/opt/splash-old"
        legacy_release.mkdir(parents=True)
        (legacy_release / "marker").write_text("preserve\n", encoding="utf-8")

        with mock.patch.object(
            splash_runtime, "_qualified_revision", side_effect=revisions
        ):
            first = splash_runtime._install_release(
                self.paths, sources[0], qualifications[0]
            )
            self.assertEqual(first["source_revision"], revisions[0])
            self.assertEqual(
                splash_runtime.verify_installation(self.paths)["verification"][
                    "status"
                ],
                "pass",
            )
            second = splash_runtime._install_release(
                self.paths, sources[1], qualifications[1]
            )
            self.assertEqual(second["source_revision"], revisions[1])
            self.assertEqual(self.paths.current.resolve().name, revisions[1])
            self.assertEqual(self.paths.previous.resolve().name, revisions[0])
            rolled_back = splash_runtime._rollback(self.paths)
            self.assertEqual(rolled_back["source_revision"], revisions[0])
            self.assertEqual(self.paths.previous.resolve().name, revisions[1])
            splash_runtime._uninstall(self.paths)

        self.assertFalse(self.paths.root.exists())
        self.assertFalse(self.paths.launcher.exists())
        self.assertEqual(legacy_shim.read_text(encoding="utf-8"), "legacy shim\n")
        self.assertEqual(
            (legacy_release / "marker").read_text(encoding="utf-8"), "preserve\n"
        )

    def test_release_inventory_drift_blocks_verify_and_uninstall(self):
        """Refuse to verify or delete a release containing an unowned entry."""
        revision = "3" * 40
        source = self.root / revision
        source.mkdir()
        self._package(source, revision)
        qualification = self._qualification(source, revision)
        with mock.patch.object(
            splash_runtime, "_qualified_revision", return_value=revision
        ):
            splash_runtime._install_release(self.paths, source, qualification)
        extra = self.paths.current.resolve() / "user-data"
        extra.write_text("keep\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            splash_runtime.verify_installation(self.paths)
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            splash_runtime._uninstall(self.paths)
        self.assertTrue(extra.exists())
        self.assertTrue(self.paths.launcher.exists())

    def test_optional_smoke_refuses_uncached_model_before_heavy_op(self):
        """Require a full local model verification before any smoke server starts."""
        revision = "5" * 40
        source = self.root / revision
        source.mkdir()
        self._package(source, revision)
        qualification = self._qualification(source, revision)
        with mock.patch.object(
            splash_runtime, "_qualified_revision", return_value=revision
        ):
            splash_runtime._install_release(self.paths, source, qualification)
        failed_verification = subprocess.CompletedProcess(
            args=["models.py", "verify"], returncode=1, stdout="", stderr="missing"
        )
        with mock.patch.object(
            splash_runtime.subprocess, "run", return_value=failed_verification
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "refusing download"):
                splash_runtime._verify_smoke(
                    self.paths,
                    "owner/repo",
                    "127.0.0.1",
                    8319,
                    30,
                    splash_runtime.HEAVY_OP_NEED_GB,
                )
        run.assert_called_once()
        self.assertIn("--full", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")

    def test_qualification_receipt_rejects_a_different_command(self):
        """Require exact command, checkout, heavy-op and output evidence bindings."""
        revision = "4" * 40
        source = self.root / revision
        source.mkdir()
        self._package(source, revision)
        receipt_path = self._qualification(source, revision)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["gates"][1]["argv"] = ["make", "check-native-cpu"]
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "command does not match"):
            splash_runtime._load_qualification(receipt_path, revision)

    def test_qualification_uses_heavy_op_and_distinct_python_stamps(self):
        """Keep every qualification target admitted and each Python environment isolated."""
        temporary = self.root / "splash-qualification-command-fixture"
        commands = splash_runtime._qualification_commands(temporary)
        self.assertEqual(
            [name for name, _, _ in commands],
            list(splash_runtime.QUALIFICATION_TARGETS),
        )
        for _, argv, environment in commands:
            self.assertEqual(argv[0], str(Path.home() / ".local/bin/heavy-op"))
            self.assertEqual(environment["HEAVY_OP_NEED_GB"], "16")
        python_scripts = [
            argv[3]
            for name, argv, _ in commands
            if name.startswith("check-python-engine:")
        ]
        self.assertEqual(len(python_scripts), 3)
        for version in ("3.12", "3.13", "3.14"):
            matching = [
                script for script in python_scripts if f"python-{version}" in script
            ]
            self.assertEqual(len(matching), 1)
            self.assertIn(f"PYTHON_CANDIDATES=python{version}", matching[0])
            self.assertIn("DEV_STAMP=", matching[0])
            self.assertIn(f'test "$actual" = "{version}"', matching[0])

    def test_install_ignores_unqualified_preexisting_dist_archive(self):
        """Install only the package copied into the qualified receipt evidence."""
        revision = "6" * 40
        source = self.root / revision
        source.mkdir()
        archive = self._package(source, revision)
        qualification = self._qualification(source, revision)
        original_hash = digest(archive)
        archive.write_bytes(b"unqualified replacement archive")
        archive.with_suffix(archive.suffix + ".sha256").write_text(
            digest(archive) + "\n", encoding="ascii"
        )

        with mock.patch.object(
            splash_runtime, "_qualified_revision", return_value=revision
        ):
            installed = splash_runtime._install_release(
                self.paths, source, qualification
            )

        package_evidence = json.loads(qualification.read_text(encoding="utf-8"))[
            "package_artifact"
        ]["archive"]
        self.assertEqual(package_evidence["sha256"], original_hash)
        self.assertEqual(installed["package"]["sha256"], original_hash)
        self.assertEqual(
            Path(installed["package"]["path"]), Path(package_evidence["path"])
        )

    def test_qualification_rejects_changed_bound_package_archive(self):
        """Refuse package bytes that no longer match their qualification hash."""
        revision = "7" * 40
        source = self.root / revision
        source.mkdir()
        self._package(source, revision)
        receipt_path = self._qualification(source, revision)
        package_path = Path(
            json.loads(receipt_path.read_text(encoding="utf-8"))["package_artifact"][
                "archive"
            ]["path"]
        )
        package_path.write_bytes(b"changed after qualification")
        with self.assertRaisesRegex(RuntimeError, "package archive evidence changed"):
            splash_runtime._load_qualification(receipt_path, revision)

    def test_archive_extraction_rejects_parent_escape(self):
        """Reject a package archive member that escapes the staging root."""
        archive_path = self.root / "unsafe.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            member = tarfile.TarInfo("release/../../outside")
            member.size = 1
            import io

            archive.addfile(member, io.BytesIO(b"x"))
        with self.assertRaisesRegex(RuntimeError, "unsafe package member"):
            splash_runtime._extract_archive(
                archive_path, self.root / "stage", "release"
            )
        self.assertFalse((self.root / "outside").exists())

    def test_stable_launcher_uses_environment_key_and_rejects_argv_key(self):
        """Keep API key material out of the stable launcher's argument vector."""
        runtime = self.root / "runtime"
        current = runtime / "current"
        (current / "python/bin").mkdir(parents=True)
        (current / "install").mkdir(parents=True)
        (current / "install/launcher.py").write_text("# fixture\n", encoding="utf-8")
        capture = self.root / "capture.json"
        python = current / "python/bin/python3"
        python.write_text(
            "#!/bin/sh\n"
            "shift 2\n"
            'python3 -c \'import json,os,sys; json.dump({"argv":sys.argv[1:],"key":os.environ.get("SPLASH_API_KEY")},open(os.environ["SPLASH_TEST_CAPTURE"],"w"))\' "$@"\n',
            encoding="utf-8",
        )
        python.chmod(0o755)
        environment = {
            **os.environ,
            "HOME": str(self.home),
            "SPLASH_RUNTIME_ROOT": str(runtime),
            "SPLASH_API_KEY": "fixture-secret",
            "SPLASH_TEST_CAPTURE": str(capture),
        }
        launcher = Path(__file__).resolve().parents[3] / "install/splash-engine"
        accepted = subprocess.run(
            [str(launcher), "--version"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        captured = json.loads(capture.read_text(encoding="utf-8"))
        self.assertEqual(captured["argv"], ["--version"])
        self.assertEqual(captured["key"], "fixture-secret")
        rejected = subprocess.run(
            [str(launcher), "--api-key", "fixture-secret"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(rejected.returncode, 64)
        self.assertNotIn("fixture-secret", rejected.stderr)
        self.assertEqual(json.loads(capture.read_text(encoding="utf-8")), captured)


if __name__ == "__main__":
    unittest.main()
