#!/usr/bin/env python3
"""Qualify, install, verify, roll back, and remove a local Splash release."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

SCHEMA_VERSION = 1
QUALIFICATION_SCHEMA = "splash-local-qualification-v1"
INSTALLATION_SCHEMA = "splash-local-installation-v1"
DEFAULT_PORT = 8318
QUALIFICATION_TARGETS = (
    "check-source",
    "check-native-cpu",
    "test-sanitizers",
    "check-python-engine:3.12",
    "check-python-engine:3.13",
    "check-python-engine:3.14",
)
HEAVY_OP_NEED_GB = 16


@dataclass(frozen=True)
class RuntimePaths:
    """User-local paths owned by the Splash release manager."""

    root: Path
    launcher: Path
    state: Path
    support: Path

    @classmethod
    def defaults(cls, home: Path | None = None) -> "RuntimePaths":
        """Return default user-local runtime and state paths."""
        resolved_home = (home or Path.home()).resolve()
        return cls(
            root=resolved_home / ".local/share/splash",
            launcher=resolved_home / ".local/bin/splash-engine",
            state=resolved_home / ".local/state/splash",
            support=resolved_home / "Library/Application Support/Splash",
        )

    @property
    def releases(self) -> Path:
        """Return the immutable release directory."""
        return self.root / "releases"

    @property
    def current(self) -> Path:
        """Return the active release link."""
        return self.root / "current"

    @property
    def previous(self) -> Path:
        """Return the rollback release link."""
        return self.root / "previous"

    @property
    def install_lock(self) -> Path:
        """Return the lock path outside the removable runtime root."""
        return self.root.with_name("splash.install.lock")

    @property
    def serve_lock(self) -> Path:
        """Return the shared runtime lock used by foreground Splash servers."""
        return self.support / "runtime/serve.lock"


def _sha256(path: Path) -> str:
    """Hash one regular file without following a final-component symlink."""
    digest = hashlib.sha256()
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"expected a regular file: {path}")
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(source.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _release_inventory(release: Path) -> dict[str, dict[str, Any]]:
    """Describe every package entry except the generated install manifest."""
    inventory: dict[str, dict[str, Any]] = {}

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(release).as_posix()
            if relative == "manifest.json":
                continue
            metadata = child.lstat()
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                inventory[relative] = {"type": "directory", "mode": mode}
                visit(child)
            elif stat.S_ISREG(metadata.st_mode):
                inventory[relative] = {
                    "type": "file",
                    "mode": mode,
                    "size": metadata.st_size,
                    "sha256": _sha256(child),
                }
            elif stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(child)
                _safe_symlink_target(PurePosixPath(relative), target)
                inventory[relative] = {
                    "type": "symlink",
                    "mode": mode,
                    "target": target,
                }
            else:
                raise RuntimeError(f"unsupported installed release entry: {child}")

    visit(release)
    return inventory


def _assert_no_symlink_components(path: Path) -> None:
    """Refuse mutation through any symlinked component of an owned path."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"refusing symlinked managed path component: {current}")
        if current.exists() and not current.is_dir() and current != absolute:
            raise RuntimeError(f"managed path component is not a directory: {current}")


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _git(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _qualified_revision(source: Path) -> str:
    """Require a clean checkout whose HEAD is the fork's fetched origin/main."""
    revision = _git(source, "rev-parse", "HEAD")
    origin_main = _git(source, "rev-parse", "refs/remotes/origin/main")
    origin_url = _git(source, "remote", "get-url", "origin").rstrip("/")
    status = _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    if origin_url not in {
        "https://github.com/jecruz/splash.git",
        "git@github.com:jecruz/splash.git",
        "ssh://git@github.com/jecruz/splash.git",
    }:
        raise RuntimeError("origin must be the jecruz/splash fork")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("source HEAD is not a full Git commit ID")
    if revision != origin_main:
        raise RuntimeError("source HEAD must equal the fetched fork origin/main")
    if status:
        raise RuntimeError("source checkout must be clean before qualification")
    return revision


def _expected_gate_argv(target: str, temporary_root: Path) -> list[str]:
    """Return the exact argv allowed for one qualification target."""
    if target == "check-source":
        return ["make", "check-source"]
    if target in {"check-native-cpu", "test-sanitizers"}:
        return [
            str(Path.home() / ".local/bin/heavy-op"),
            "make",
            target,
            "PYTHON_CANDIDATES=python3",
        ]
    match = re.fullmatch(r"check-python-engine:(3\.(?:12|13|14))", target)
    if not match:
        raise RuntimeError(f"unknown local qualification target: {target}")
    version = match.group(1)
    return [
        str(Path.home() / ".local/bin/heavy-op"),
        "make",
        "check-python-engine",
        f"PYTHON_CANDIDATES=python{version}",
        f"VENV={temporary_root / f'python-{version}'}",
    ]


def _run_recorded(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except BaseException:
        os.close(stdout_fd)
        raise
    with os.fdopen(stdout_fd, "wb") as stdout, os.fdopen(stderr_fd, "wb") as stderr:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "argv": list(command),
        "cwd": str(cwd),
        "return_code": result.returncode,
        "heavy_op_need_gb": (
            int(environment["HEAVY_OP_NEED_GB"])
            if environment and "HEAVY_OP_NEED_GB" in environment
            else None
        ),
        "stdout": {
            "path": str(stdout_path),
            "size": stdout_path.stat().st_size,
            "sha256": _sha256(stdout_path),
        },
        "stderr": {
            "path": str(stderr_path),
            "size": stderr_path.stat().st_size,
            "sha256": _sha256(stderr_path),
        },
    }


def _heavy_command(
    command: Sequence[str], *, need_gb: int
) -> tuple[list[str], dict[str, str]]:
    """Return the required heavy-op argv and its admission environment."""
    heavy_op = Path.home() / ".local/bin/heavy-op"
    if not heavy_op.is_file() or not os.access(heavy_op, os.X_OK):
        raise RuntimeError(f"required heavy-op executable is unavailable: {heavy_op}")
    environment = os.environ.copy()
    environment["HEAVY_OP_NEED_GB"] = str(need_gb)
    return [str(heavy_op), *command], environment


def _qualification_commands(
    temporary: Path,
) -> list[tuple[str, list[str], dict[str, str] | None]]:
    """Construct the approved local-CI targets and fresh Python environments."""
    commands: list[tuple[str, list[str], dict[str, str] | None]] = [
        ("check-source", ["make", "check-source"], None)
    ]
    for target in ("check-native-cpu", "test-sanitizers"):
        heavy_argv, heavy_env = _heavy_command(
            ["make", target, "PYTHON_CANDIDATES=python3"], need_gb=HEAVY_OP_NEED_GB
        )
        commands.append((target, heavy_argv, heavy_env))
    for version in ("3.12", "3.13", "3.14"):
        candidate = f"python{version}"
        venv = temporary / f"python-{version}"
        heavy_argv, heavy_env = _heavy_command(
            [
                "make",
                "check-python-engine",
                f"PYTHON_CANDIDATES={candidate}",
                f"VENV={venv}",
            ],
            need_gb=HEAVY_OP_NEED_GB,
        )
        commands.append((f"check-python-engine:{version}", heavy_argv, heavy_env))
    return commands


def qualify_source(source: Path, state_root: Path) -> dict[str, Any]:
    """Run the local-CI-equivalent gates for a clean fork origin/main commit."""
    source = source.resolve()
    state_root = state_root.expanduser().resolve()
    revision = _qualified_revision(source)
    attempt = (
        state_root
        / "qualifications"
        / revision
        / f"{_utc_now().replace(':', '')}-{uuid.uuid4().hex[:8]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    gates: list[dict[str, Any]] = []
    overall = "pass"
    with tempfile.TemporaryDirectory(prefix="splash-qualification-") as temporary_name:
        temporary = Path(temporary_name)
        for index, (name, command, environment) in enumerate(
            _qualification_commands(temporary), start=1
        ):
            record = _run_recorded(
                command,
                cwd=source,
                stdout_path=attempt / f"{index:02d}-{name.replace(':', '-')}.stdout",
                stderr_path=attempt / f"{index:02d}-{name.replace(':', '-')}.stderr",
                environment=environment,
            )
            record["target"] = name
            record["temporary_root"] = str(temporary)
            record["status"] = "pass" if record["return_code"] == 0 else "fail"
            gates.append(record)
            if record["return_code"] != 0:
                overall = "fail"
                break
    try:
        final_revision = _qualified_revision(source)
        source_recheck = {"status": "pass", "revision": final_revision}
        if final_revision != revision:
            overall = "fail"
            source_recheck["status"] = "fail"
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        source_recheck = {"status": "fail", "error": str(error)}
        overall = "fail"
    receipt = {
        "schema": QUALIFICATION_SCHEMA,
        "status": overall,
        "source_revision": revision,
        "source_remote": "origin",
        "source_ref": "refs/remotes/origin/main",
        "qualified_at": _utc_now(),
        "qualification_source": "local-ci-equivalent",
        "source_path": str(source),
        "origin_url": _git(source, "remote", "get-url", "origin").rstrip("/"),
        "source_recheck": source_recheck,
        "hosted_ci": {"provider": "woodpecker", "status": "not-configured"},
        "heavy_op": {"need_gb": HEAVY_OP_NEED_GB},
        "required_targets": list(QUALIFICATION_TARGETS),
        "gates": gates,
    }
    receipt_path = attempt / "qualification.json"
    _write_json_exclusive(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = _sha256(receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if overall != "pass":
        raise RuntimeError(f"local qualification failed; evidence: {receipt_path}")
    if len(gates) != len(QUALIFICATION_TARGETS):
        raise RuntimeError("local qualification did not execute every required target")
    return receipt


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Create a JSON file once, without replacing existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _read_regular(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one bounded regular file through a no-follow descriptor."""
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise RuntimeError(f"expected a bounded regular file: {path}")
        raw = source.read(maximum_bytes + 1)
        after = os.fstat(source.fileno())
        if len(raw) > maximum_bytes or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"file changed while reading: {path}")
        return raw


def _load_qualification(path: Path, revision: str) -> tuple[dict[str, Any], str]:
    """Validate a successful receipt bound to the exact source revision."""
    try:
        raw = _read_regular(path, maximum_bytes=4 * 1024 * 1024)
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read qualification receipt: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError("qualification receipt must be a JSON object")
    if (
        document.get("schema") != QUALIFICATION_SCHEMA
        or document.get("status") != "pass"
        or document.get("source_revision") != revision
        or document.get("source_remote") != "origin"
        or document.get("source_ref") != "refs/remotes/origin/main"
        or document.get("qualification_source") != "local-ci-equivalent"
        or document.get("source_recheck") != {"status": "pass", "revision": revision}
        or document.get("hosted_ci")
        != {"provider": "woodpecker", "status": "not-configured"}
        or document.get("heavy_op") != {"need_gb": HEAVY_OP_NEED_GB}
        or document.get("required_targets") != list(QUALIFICATION_TARGETS)
    ):
        raise RuntimeError(
            "qualification receipt does not match the required local gate"
        )
    source_path_text = document.get("source_path")
    if (
        not isinstance(source_path_text, str)
        or not Path(source_path_text).is_absolute()
    ):
        raise RuntimeError("qualification receipt has no absolute source path")
    source_path = Path(source_path_text)
    if document.get("origin_url") not in {
        "https://github.com/jecruz/splash.git",
        "git@github.com:jecruz/splash.git",
        "ssh://git@github.com/jecruz/splash.git",
    }:
        raise RuntimeError("qualification receipt is not bound to the fork")
    gates = document.get("gates")
    if not isinstance(gates, list) or any(not isinstance(gate, dict) for gate in gates):
        raise RuntimeError("qualification receipt has invalid gate records")
    if [gate.get("target") for gate in gates] != list(QUALIFICATION_TARGETS):
        raise RuntimeError("qualification receipt is missing required gate records")
    python_venvs: set[Path] = set()
    temporary_roots: set[Path] = set()
    for gate in gates:
        if (
            not isinstance(gate, dict)
            or gate.get("status") != "pass"
            or gate.get("return_code") != 0
        ):
            raise RuntimeError("qualification receipt contains a failed gate")
        target = gate.get("target")
        temporary_text = gate.get("temporary_root")
        if (
            not isinstance(temporary_text, str)
            or not Path(temporary_text).is_absolute()
        ):
            raise RuntimeError(f"qualification target {target} has no temporary root")
        temporary_root = Path(temporary_text)
        if not temporary_root.name.startswith("splash-qualification-"):
            raise RuntimeError(
                "Python qualification did not use a fresh temporary root"
            )
        temporary_roots.add(temporary_root)
        if gate.get("cwd") != str(source_path):
            raise RuntimeError("qualification command ran from a different source tree")
        if gate.get("argv") != _expected_gate_argv(target, temporary_root):
            raise RuntimeError(f"qualification command does not match target {target}")
        expected_need = None if target == "check-source" else HEAVY_OP_NEED_GB
        if gate.get("heavy_op_need_gb") != expected_need:
            raise RuntimeError(
                f"qualification target {target} used the wrong heavy-op policy"
            )
        if target.startswith("check-python-engine:"):
            venv_path = temporary_root / f"python-{target.split(':', 1)[1]}"
            if venv_path in python_venvs:
                raise RuntimeError("Python qualification reused a virtual environment")
            python_venvs.add(venv_path)
        for key in ("stdout", "stderr"):
            stream = gate.get(key)
            if not isinstance(stream, dict):
                raise RuntimeError(f"qualification receipt is missing {key} evidence")
            stream_path = Path(stream.get("path", ""))
            if (
                not stream_path.is_relative_to(path.parent)
                or stream_path.is_symlink()
                or not stream_path.is_file()
                or stream_path.stat().st_size != stream.get("size")
                or _sha256(stream_path) != stream.get("sha256")
            ):
                raise RuntimeError(f"qualification {key} evidence changed")
    if len(temporary_roots) != 1 or len(python_venvs) != 3:
        raise RuntimeError(
            "qualification did not use one fresh root and three Python environments"
        )
    return document, hashlib.sha256(raw).hexdigest()


def _safe_relative(name: str) -> PurePosixPath:
    """Validate an archive member path before touching the staging tree."""
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or path.as_posix() != name.rstrip("/")
        or any(part in {"", "."} for part in path.parts)
    ):
        raise RuntimeError(f"unsafe package member path: {name!r}")
    return path


def _safe_symlink_target(member_path: PurePosixPath, target: str) -> None:
    """Require an archive symlink target to remain lexically inside its release."""
    target_path = PurePosixPath(target)
    if not target or target_path.is_absolute() or "\\" in target:
        raise RuntimeError(f"unsafe package symlink target: {target!r}")
    resolved = PurePosixPath(posixpath.normpath(str(member_path.parent / target_path)))
    if resolved.is_absolute() or ".." in resolved.parts:
        raise RuntimeError(
            f"package symlink escapes release: {member_path} -> {target}"
        )


def _ensure_real_parents(root: Path, relative: PurePosixPath) -> Path:
    """Require already-created path parents to be real directories."""
    parent = root
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.is_symlink() or not parent.is_dir():
            raise RuntimeError(f"package path has an unsafe parent: {parent}")
    return root.joinpath(*relative.parts)


def _extract_archive(archive_path: Path, staging: Path, expected_top: str) -> Path:
    """Extract only regular files, directories, and contained relative symlinks."""
    staging.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            directories: list[tuple[Path, int]] = []
            for member in members:
                relative = _safe_relative(member.name)
                if relative.parts[0] != expected_top:
                    raise RuntimeError(
                        "package archive has an unexpected top-level path"
                    )
                destination = _ensure_real_parents(staging, relative)
                if member.isdir():
                    if destination.exists() or destination.is_symlink():
                        raise RuntimeError(f"duplicate package path: {member.name}")
                    destination.mkdir()
                    directories.append((destination, member.mode & 0o777))
                elif member.isfile():
                    if member.mode & 0o7000:
                        raise RuntimeError(f"unsafe package file mode: {member.name}")
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(
                            f"package file has no content: {member.name}"
                        )
                    descriptor = os.open(
                        destination,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                    with source, os.fdopen(descriptor, "wb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    destination.chmod(member.mode & 0o777)
                elif member.issym():
                    _safe_symlink_target(relative, member.linkname)
                    if destination.exists() or destination.is_symlink():
                        raise RuntimeError(f"duplicate package path: {member.name}")
                    destination.symlink_to(member.linkname)
                else:
                    raise RuntimeError(
                        f"unsupported package member type: {member.name}"
                    )
            for directory, mode in sorted(
                directories, key=lambda item: len(item[0].parts), reverse=True
            ):
                directory.chmod(mode)
            top = staging / expected_top
            if top.is_symlink() or not top.is_dir():
                raise RuntimeError("package archive is missing its release directory")
            return top
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _atomic_symlink(target: Path, link: Path) -> None:
    """Atomically replace a managed release link with a relative symlink."""
    _assert_no_symlink_components(link.parent)
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}.tmp")
    relative = os.path.relpath(target, start=link.parent)
    temporary.symlink_to(relative)
    try:
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _mutation_lock(paths: RuntimePaths) -> Iterator[None]:
    """Serialize installer operations and exclude active Splash servers."""
    for path in (paths.root, paths.launcher, paths.state, paths.support):
        _assert_no_symlink_components(path)
    paths.install_lock.parent.mkdir(parents=True, exist_ok=True)
    if paths.install_lock.is_symlink():
        raise RuntimeError(f"refusing symlinked installer lock: {paths.install_lock}")
    install_fd = os.open(
        paths.install_lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(install_fd, "a+") as install_lock:
        if not stat.S_ISREG(os.fstat(install_lock.fileno()).st_mode):
            raise RuntimeError(
                f"installer lock is not a regular file: {paths.install_lock}"
            )
        try:
            fcntl.flock(install_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another Splash installation operation is active"
            ) from error
        try:
            paths.serve_lock.parent.mkdir(parents=True, exist_ok=True)
            if paths.serve_lock.is_symlink():
                raise RuntimeError(
                    f"refusing symlinked server lock: {paths.serve_lock}"
                )
            serve_fd = os.open(
                paths.serve_lock,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(serve_fd, "a+") as serve_lock:
                if not stat.S_ISREG(os.fstat(serve_lock.fileno()).st_mode):
                    raise RuntimeError(
                        f"server lock is not a regular file: {paths.serve_lock}"
                    )
                try:
                    fcntl.flock(serve_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(
                        "a Splash server is active; stop it before changing releases"
                    ) from error
                try:
                    yield
                finally:
                    fcntl.flock(serve_lock.fileno(), fcntl.LOCK_UN)
        finally:
            fcntl.flock(install_lock.fileno(), fcntl.LOCK_UN)


def _read_release(release: Path) -> dict[str, Any]:
    """Read the embedded package identity and verify its native artifacts."""
    metadata_path = release / "release.json"
    try:
        metadata = json.loads(_read_regular(metadata_path, maximum_bytes=1024 * 1024))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid package release metadata: {metadata_path}"
        ) from error
    if not isinstance(metadata, dict):
        raise RuntimeError("package release metadata must be a JSON object")
    for relative, key in (
        ("engine/splash", "binary_sha256"),
        ("engine/splash.metallib", "metallib_sha256"),
    ):
        artifact = release / relative
        expected = metadata.get(key)
        if (
            not artifact.is_file()
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or _sha256(artifact) != expected
        ):
            raise RuntimeError(f"installed artifact hash mismatch: {relative}")
    python = release / "python/bin/python3"
    launcher = release / "install/launcher.py"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("package is missing its executable embedded Python runtime")
    if not launcher.is_file():
        raise RuntimeError("package is missing install/launcher.py")
    if not isinstance(metadata.get("version"), str) or not metadata["version"]:
        raise RuntimeError("package release version is invalid")
    return metadata


def _validate_manifest(paths: RuntimePaths, release: Path) -> dict[str, Any]:
    """Validate the local install manifest and all release identity bindings."""
    _assert_no_symlink_components(paths.releases)
    manifest_path = release / "manifest.json"
    try:
        manifest = json.loads(
            _read_regular(manifest_path, maximum_bytes=4 * 1024 * 1024)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid Splash install manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("schema") != INSTALLATION_SCHEMA:
        raise RuntimeError("unsupported Splash install manifest schema")
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("manifest source_revision is not a full Git commit ID")
    if (
        release.is_symlink()
        or release.name != revision
        or release.parent.resolve() != paths.releases.resolve()
        or manifest.get("source_remote") != "origin"
        or manifest.get("source_ref") != "refs/remotes/origin/main"
    ):
        raise RuntimeError("release directory does not match its source revision")
    if manifest.get("release_path") != str(release.resolve()):
        raise RuntimeError("manifest release_path does not match release directory")
    archive = manifest.get("package")
    if not isinstance(archive, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(archive.get("sha256", ""))
    ):
        raise RuntimeError("manifest package identity is invalid")
    package_metadata = _read_release(release)
    if manifest.get("version") != package_metadata.get("version"):
        raise RuntimeError("manifest package version does not match release.json")
    if manifest.get("default_port") != DEFAULT_PORT:
        raise RuntimeError(
            "manifest default port does not match this installer contract"
        )
    launcher = manifest.get("launcher")
    if (
        not isinstance(launcher, dict)
        or launcher.get("path") != str(paths.launcher)
        or not re.fullmatch(r"[0-9a-f]{64}", str(launcher.get("template_sha256", "")))
    ):
        raise RuntimeError("manifest launcher identity is invalid")
    qualification = manifest.get("qualification")
    if not isinstance(qualification, dict):
        raise RuntimeError("manifest qualification is missing")
    receipt_path = Path(str(qualification.get("receipt_path", "")))
    receipt_hash = qualification.get("receipt_sha256")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(receipt_hash))
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or _sha256(receipt_path) != receipt_hash
    ):
        raise RuntimeError("manifest qualification receipt is missing or changed")
    receipt, _ = _load_qualification(receipt_path, revision)
    if manifest.get("source_path") != receipt["source_path"]:
        raise RuntimeError("manifest source path does not match qualification receipt")
    if manifest.get("origin_url") != receipt["origin_url"]:
        raise RuntimeError("manifest origin URL does not match qualification receipt")
    expected_qualification = {
        "source": receipt["qualification_source"],
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_hash,
        "targets": receipt["required_targets"],
        "hosted_ci": receipt["hosted_ci"],
    }
    if qualification != expected_qualification:
        raise RuntimeError("manifest qualification does not match its receipt")
    if manifest.get("binary_sha256") != _sha256(release / "engine/splash"):
        raise RuntimeError("manifest engine binary hash mismatch")
    if manifest.get("metallib_sha256") != _sha256(release / "engine/splash.metallib"):
        raise RuntimeError("manifest metallib hash mismatch")
    if manifest.get("package_files") != _release_inventory(release):
        raise RuntimeError("installed package file inventory changed")
    return manifest


def _current_release(paths: RuntimePaths, link: Path) -> Path | None:
    """Resolve and validate a managed release symlink, if present."""
    if not link.is_symlink():
        if link.exists():
            raise RuntimeError(f"managed release link is not a symlink: {link}")
        return None
    release = link.resolve(strict=True)
    _validate_manifest(paths, release)
    return release


def _launcher_bytes(release: Path) -> bytes:
    """Read the packaged stable launcher template."""
    path = release / "install/splash-engine"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("package is missing the regular stable launcher template")
    content = _read_regular(path, maximum_bytes=1024 * 1024)
    if b".local/share/splash" not in content or b"/current" not in content:
        raise RuntimeError(
            "stable launcher template does not target the managed runtime"
        )
    return content


def _install_launcher(paths: RuntimePaths, release: Path) -> str:
    """Install the stable launcher only if an existing file is owned by Splash."""
    content = _launcher_bytes(release)
    expected_hash = hashlib.sha256(content).hexdigest()
    launcher = paths.launcher
    launcher.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(launcher.parent)
    if launcher.exists() or launcher.is_symlink():
        if launcher.is_symlink() or not launcher.is_file():
            raise RuntimeError(
                f"refusing to replace non-regular stable launcher: {launcher}"
            )
        actual_hash = _sha256(launcher)
        if actual_hash != expected_hash:
            owned = False
            for link in (paths.current, paths.previous):
                existing = _current_release(paths, link)
                if (
                    existing
                    and _sha256(existing / "install/splash-engine") == actual_hash
                ):
                    owned = True
                    break
            if not owned:
                raise RuntimeError(
                    f"refusing to replace unowned stable launcher: {launcher}"
                )
    temporary = launcher.with_name(f".{launcher.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o755)
        os.replace(temporary, launcher)
    finally:
        temporary.unlink(missing_ok=True)
    return expected_hash


def _restore_launcher(paths: RuntimePaths, previous: bytes | None) -> None:
    """Restore the stable launcher after a failed release promotion."""
    if previous is None:
        paths.launcher.unlink(missing_ok=True)
        return
    temporary = paths.launcher.with_name(
        f".{paths.launcher.name}.{uuid.uuid4().hex}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
    with os.fdopen(descriptor, "wb") as output:
        output.write(previous)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(temporary, 0o755)
    os.replace(temporary, paths.launcher)


def _source_version(source: Path, revision: str) -> str:
    """Return a unique package version tied to the exact main commit."""
    return f"local-{revision}"


def _build_package(source: Path, revision: str) -> Path:
    """Build one release archive through the mandatory heavy-op wrapper."""
    _assert_no_symlink_components(source / "dist")
    if (source / "dist").exists() and not (source / "dist").is_dir():
        raise RuntimeError("package output path is not a directory")
    version = _source_version(source, revision)
    heavy_argv, environment = _heavy_command(
        ["make", "package", f"RELEASE_VERSION={version}", "PYTHON_CANDIDATES=python3"],
        need_gb=HEAVY_OP_NEED_GB,
    )
    subprocess.run(heavy_argv, cwd=source, env=environment, check=True)
    archive = source / "dist" / f"splash-{version}-arm64-macos26.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    if not archive.is_file() or not checksum.is_file():
        raise RuntimeError(
            "heavy-op package command did not produce archive and checksum"
        )
    expected = _read_regular(checksum, maximum_bytes=128).decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or _sha256(archive) != expected:
        raise RuntimeError("built package checksum does not match its sidecar")
    return archive


def _install_release(
    paths: RuntimePaths,
    source: Path,
    qualification_path: Path,
) -> dict[str, Any]:
    """Build and atomically promote a qualified fork-main release."""
    source = source.resolve()
    revision = _qualified_revision(source)
    qualification_path = qualification_path.expanduser().absolute()
    if qualification_path.is_symlink() or not qualification_path.is_file():
        raise RuntimeError("qualification receipt must be a regular non-symlink file")
    qualification_path = qualification_path.resolve(strict=True)
    qualification, qualification_hash = _load_qualification(
        qualification_path, revision
    )
    if qualification["source_path"] != str(source):
        raise RuntimeError("qualification receipt belongs to another source checkout")
    destination = paths.releases / revision
    with _mutation_lock(paths):
        old_current = _current_release(paths, paths.current)
        old_previous = _current_release(paths, paths.previous)
        if paths.launcher.is_symlink() or (
            paths.launcher.exists() and not paths.launcher.is_file()
        ):
            raise RuntimeError("refusing an unsafe stable launcher path")
        old_launcher = (
            _read_regular(paths.launcher, maximum_bytes=1024 * 1024)
            if paths.launcher.is_file()
            else None
        )
        paths.releases.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink_components(source / "dist")
        if (source / "dist").exists() and not (source / "dist").is_dir():
            raise RuntimeError("package output path is not a directory")
        version = _source_version(source, revision)
        archive_path = source / "dist" / f"splash-{version}-arm64-macos26.tar.gz"
        checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
        if archive_path.is_symlink():
            raise RuntimeError(f"refusing symlinked package archive: {archive_path}")
        if not archive_path.exists():
            archive_path = _build_package(source, revision)
            checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
        archive_hash = _sha256(archive_path)
        if (
            not checksum_path.is_file()
            or _read_regular(checksum_path, maximum_bytes=128).decode("ascii").strip()
            != archive_hash
        ):
            raise RuntimeError(
                "package archive checksum sidecar is missing or incorrect"
            )
        if destination.is_symlink():
            raise RuntimeError(f"refusing symlinked release destination: {destination}")
        if destination.exists():
            existing = _validate_manifest(paths, destination)
            if existing.get("package", {}).get("sha256") != archive_hash:
                raise RuntimeError(
                    "immutable release already exists with different package bytes"
                )
            if old_current == destination.resolve():
                launcher_updated = False
                try:
                    _install_launcher(paths, destination)
                    launcher_updated = True
                    return verify_installation(paths)
                except BaseException:
                    if launcher_updated:
                        _restore_launcher(paths, old_launcher)
                    raise
        else:
            staging = paths.releases / f".staging-{uuid.uuid4().hex}"
            version = _source_version(source, revision)
            staged_root = _extract_archive(
                archive_path, staging, f"splash-{version}-arm64-macos26"
            )
            metadata = _read_release(staged_root)
            if metadata["version"] != version:
                shutil.rmtree(staging)
                raise RuntimeError(
                    "package release version does not match qualified source"
                )
            manifest = {
                "schema": INSTALLATION_SCHEMA,
                "source_revision": revision,
                "source_remote": "origin",
                "source_ref": "refs/remotes/origin/main",
                "source_path": qualification["source_path"],
                "origin_url": qualification["origin_url"],
                "qualified_at": qualification.get("qualified_at"),
                "qualification": {
                    "source": qualification["qualification_source"],
                    "status": qualification["status"],
                    "receipt_path": str(qualification_path),
                    "receipt_sha256": qualification_hash,
                    "targets": qualification["required_targets"],
                    "hosted_ci": qualification["hosted_ci"],
                },
                "version": metadata["version"],
                "installed_at": _utc_now(),
                "release_path": str(destination.resolve()),
                "package": {"path": str(archive_path), "sha256": archive_hash},
                "binary_sha256": metadata["binary_sha256"],
                "metallib_sha256": metadata["metallib_sha256"],
                "python": {
                    "path": str(destination / "python/bin/python3"),
                    "version": _python_version(staged_root / "python/bin/python3"),
                },
                "launcher": {
                    "path": str(paths.launcher),
                    "template_sha256": hashlib.sha256(
                        _launcher_bytes(staged_root)
                    ).hexdigest(),
                },
                "default_port": DEFAULT_PORT,
                "package_files": _release_inventory(staged_root),
            }
            _write_json_exclusive(staged_root / "manifest.json", manifest)
            # The archive's top-level folder is nested beneath its private staging root.
            staged_release = staged_root
            if destination.exists():
                raise RuntimeError(
                    f"release destination appeared during install: {destination}"
                )
            os.replace(staged_release, destination)
            shutil.rmtree(staging, ignore_errors=True)
            _validate_manifest(paths, destination)
        launcher_updated = False
        try:
            launcher_hash = _install_launcher(paths, destination)
            launcher_updated = True
            if old_current is not None and old_current != destination.resolve():
                _atomic_symlink(old_current, paths.previous)
            _atomic_symlink(destination, paths.current)
            installed = verify_installation(paths)
            installed["launcher"]["installed_sha256"] = launcher_hash
            return installed
        except BaseException:
            if old_current is not None:
                _atomic_symlink(old_current, paths.current)
            else:
                paths.current.unlink(missing_ok=True)
            if old_previous is not None:
                _atomic_symlink(old_previous, paths.previous)
            else:
                paths.previous.unlink(missing_ok=True)
            if launcher_updated:
                _restore_launcher(paths, old_launcher)
            raise


def _python_version(python: Path) -> str:
    """Read the embedded Python version without importing Splash modules."""
    result = subprocess.run(
        [str(python), "-I", "-c", "import platform; print(platform.python_version())"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _smoke_child(
    paths: RuntimePaths, model_id: str, host: str, port: int, timeout: float
) -> None:
    """Start one already-installed model and stop it after an HTTP health check."""
    import time
    from urllib.error import URLError
    from urllib.request import urlopen

    command = [
        str(paths.launcher),
        "serve",
        "--model",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
    ]
    environment = os.environ.copy()
    environment["SPLASH_RUNTIME_ROOT"] = str(paths.root)
    environment["HF_HUB_OFFLINE"] = "1"
    process = subprocess.Popen(command, cwd=tempfile.gettempdir(), env=environment)
    try:
        deadline = time.monotonic() + timeout
        endpoint = f"http://{host}:{port}/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Splash exited before health smoke (status {process.returncode})"
                )
            try:
                with urlopen(endpoint, timeout=1.0) as response:
                    payload = json.loads(response.read())
                    if response.status == 200 and payload.get("status") == "ok":
                        print(
                            "SPLASH_SMOKE_RESULT="
                            + json.dumps(
                                {
                                    "status": "pass",
                                    "endpoint": endpoint,
                                    "model": model_id,
                                },
                                sort_keys=True,
                            )
                        )
                        return
            except (OSError, ValueError, URLError):
                time.sleep(0.25)
        raise RuntimeError(f"timed out waiting for health endpoint: {endpoint}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _verify_smoke(
    paths: RuntimePaths,
    model_id: str,
    host: str,
    port: int,
    timeout: float,
    need_gb: int,
) -> dict[str, Any]:
    """Run an optional, cached-model health smoke inside heavy-op."""
    if not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9._-]*/[A-Za-z0-9_][A-Za-z0-9._-]*", model_id
    ):
        raise RuntimeError("smoke model must be a full owner/repository ID")
    release = _current_release(paths, paths.current)
    if release is None:
        raise RuntimeError("cannot smoke test without a verified active release")
    model_check = subprocess.run(
        [
            str(release / "python/bin/python3"),
            "-B",
            str(release / "install/models.py"),
            "--models",
            str(paths.support / "models"),
            "--model",
            model_id,
            "verify",
            "--full",
        ],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    if model_check.returncode != 0:
        raise RuntimeError(
            f"smoke model is not already fully verified; refusing download: {model_id}"
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_smoke-child",
        "--runtime-root",
        str(paths.root),
        "--launcher",
        str(paths.launcher),
        "--state-root",
        str(paths.state),
        "--support-root",
        str(paths.support),
        "--model",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--timeout",
        str(timeout),
    ]
    heavy_argv, environment = _heavy_command(command, need_gb=need_gb)
    result = subprocess.run(
        heavy_argv, env=environment, capture_output=True, text=True, check=True
    )
    marker = next(
        (
            line.removeprefix("SPLASH_SMOKE_RESULT=")
            for line in reversed(result.stdout.splitlines())
            if line.startswith("SPLASH_SMOKE_RESULT=")
        ),
        None,
    )
    if marker is None:
        raise RuntimeError("heavy-op health smoke returned without a health result")
    return json.loads(marker)


def verify_installation(
    paths: RuntimePaths,
    *,
    smoke_model: str | None = None,
    smoke_host: str = "127.0.0.1",
    smoke_port: int = 8319,
    timeout: float = 120,
    smoke_need_gb: int = HEAVY_OP_NEED_GB,
) -> dict[str, Any]:
    """Verify active links, manifest, package hashes, launcher, and optional health."""
    release = _current_release(paths, paths.current)
    if release is None:
        raise RuntimeError(f"no active Splash release: {paths.current}")
    manifest = _validate_manifest(paths, release)
    if not paths.launcher.is_file() or paths.launcher.is_symlink():
        raise RuntimeError(f"stable launcher is missing or unsafe: {paths.launcher}")
    launcher_hash = _sha256(paths.launcher)
    if launcher_hash != manifest["launcher"]["template_sha256"]:
        raise RuntimeError("stable launcher hash does not match release manifest")
    version = subprocess.run(
        [str(paths.launcher), "--version"],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "SPLASH_RUNTIME_ROOT": str(paths.root)},
    ).stdout.strip()
    package_metadata = _read_release(release)
    expected_version = f"Splash {package_metadata['version']}"
    if version != expected_version:
        raise RuntimeError(
            f"launcher version mismatch: {version!r}, expected {expected_version!r}"
        )
    result = dict(manifest)
    result["verification"] = {
        "status": "pass",
        "active_release": str(release),
        "launcher_sha256": launcher_hash,
        "version_output": version,
    }
    if smoke_model:
        result["verification"]["health_smoke"] = _verify_smoke(
            paths, smoke_model, smoke_host, smoke_port, timeout, smoke_need_gb
        )
    return result


def _rollback(paths: RuntimePaths) -> dict[str, Any]:
    """Atomically swap validated current and previous releases."""
    with _mutation_lock(paths):
        current = _current_release(paths, paths.current)
        previous = _current_release(paths, paths.previous)
        if previous is None:
            raise RuntimeError("no verified previous Splash release is available")
        old_launcher = _read_regular(paths.launcher, maximum_bytes=1024 * 1024)
        launcher_updated = False
        try:
            _install_launcher(paths, previous)
            launcher_updated = True
            _atomic_symlink(previous, paths.current)
            if current:
                _atomic_symlink(current, paths.previous)
            else:
                paths.previous.unlink(missing_ok=True)
            return verify_installation(paths)
        except BaseException:
            if current:
                _atomic_symlink(current, paths.current)
            else:
                paths.current.unlink(missing_ok=True)
            if previous:
                _atomic_symlink(previous, paths.previous)
            if launcher_updated:
                _restore_launcher(paths, old_launcher)
            raise


def _uninstall(paths: RuntimePaths) -> None:
    """Remove only a verified Splash runtime and its owned stable launcher."""
    with _mutation_lock(paths):
        if not paths.root.exists():
            if paths.launcher.exists():
                raise RuntimeError(
                    "Splash runtime is absent but stable launcher remains unowned"
                )
            return
        current = _current_release(paths, paths.current)
        if current is None:
            raise RuntimeError(
                "refusing to remove runtime without a verified current release"
            )
        _current_release(paths, paths.previous)
        manifest = _validate_manifest(paths, current)
        if (
            not paths.launcher.is_file()
            or _sha256(paths.launcher) != manifest["launcher"]["template_sha256"]
        ):
            raise RuntimeError("refusing to remove an unowned stable launcher")
        allowed = {"releases", "current", "previous"}
        unexpected = sorted(
            entry.name for entry in paths.root.iterdir() if entry.name not in allowed
        )
        if unexpected:
            raise RuntimeError(
                f"refusing to remove unowned runtime entries: {unexpected}"
            )
        if paths.releases.is_symlink() or not paths.releases.is_dir():
            raise RuntimeError("refusing to remove an unsafe releases directory")
        for release in paths.releases.iterdir():
            if release.is_symlink() or not release.is_dir():
                raise RuntimeError(
                    f"refusing to remove an unowned release entry: {release}"
                )
            _validate_manifest(paths, release)
        paths.launcher.unlink()
        shutil.rmtree(paths.root)


def build_parser() -> argparse.ArgumentParser:
    """Build the local Splash lifecycle CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name in ("qualify", "install", "update"):
        command = subparsers.add_parser(command_name)
        command.add_argument(
            "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
        )
        if command_name == "qualify":
            command.add_argument(
                "--state-root", type=Path, default=RuntimePaths.defaults().state
            )
        else:
            command.add_argument("--qualification", type=Path, required=True)
            command.add_argument(
                "--runtime-root", type=Path, default=RuntimePaths.defaults().root
            )
            command.add_argument(
                "--launcher", type=Path, default=RuntimePaths.defaults().launcher
            )
            command.add_argument(
                "--state-root", type=Path, default=RuntimePaths.defaults().state
            )
            command.add_argument(
                "--support-root", type=Path, default=RuntimePaths.defaults().support
            )
    for command_name in ("verify", "rollback", "uninstall"):
        command = subparsers.add_parser(command_name)
        command.add_argument(
            "--runtime-root", type=Path, default=RuntimePaths.defaults().root
        )
        command.add_argument(
            "--launcher", type=Path, default=RuntimePaths.defaults().launcher
        )
        command.add_argument(
            "--state-root", type=Path, default=RuntimePaths.defaults().state
        )
        command.add_argument(
            "--support-root", type=Path, default=RuntimePaths.defaults().support
        )
        if command_name == "verify":
            command.add_argument("--smoke-model")
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8319)
            command.add_argument("--timeout", type=float, default=120)
            command.add_argument(
                "--smoke-need-gb",
                type=int,
                default=HEAVY_OP_NEED_GB,
                help="heavy-op memory reservation for the optional model health smoke",
            )
    smoke = subparsers.add_parser("_smoke-child", help=argparse.SUPPRESS)
    smoke.add_argument("--runtime-root", type=Path, required=True)
    smoke.add_argument("--launcher", type=Path, required=True)
    smoke.add_argument("--state-root", type=Path, required=True)
    smoke.add_argument("--support-root", type=Path, required=True)
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--host", required=True)
    smoke.add_argument("--port", type=int, required=True)
    smoke.add_argument("--timeout", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one Splash qualification or local lifecycle operation."""
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "qualify":
            qualify_source(arguments.source_root, arguments.state_root)
            return 0
        paths = RuntimePaths(
            root=arguments.runtime_root.expanduser().absolute(),
            launcher=arguments.launcher.expanduser().absolute(),
            state=arguments.state_root.expanduser().absolute(),
            support=arguments.support_root.expanduser().absolute(),
        )
        if arguments.command == "_smoke-child":
            _smoke_child(
                paths,
                arguments.model,
                arguments.host,
                arguments.port,
                arguments.timeout,
            )
            return 0
        if arguments.command in {"install", "update"}:
            result = _install_release(
                paths,
                arguments.source_root,
                arguments.qualification,
            )
        elif arguments.command == "verify":
            result = verify_installation(
                paths,
                smoke_model=arguments.smoke_model,
                smoke_host=arguments.host,
                smoke_port=arguments.port,
                timeout=arguments.timeout,
                smoke_need_gb=arguments.smoke_need_gb,
            )
        elif arguments.command == "rollback":
            result = _rollback(paths)
        else:
            _uninstall(paths)
            result = {"status": "uninstalled", "runtime_root": str(paths.root)}
    except (
        OSError,
        RuntimeError,
        ValueError,
        tarfile.TarError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        print(
            json.dumps({"status": "error", "error": str(error)}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
