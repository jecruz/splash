#!/usr/bin/env python3
"""Build the runtime-only macOS archive and its Homebrew formula."""

import argparse
import contextlib
import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260825/"
    "cpython-3.13.15%2B20260825-aarch64-apple-darwin-install_only_stripped.tar.gz"
)
PYTHON_SHA256 = "149038dd0c194c25d4616d7e42a35f67f2edee96412788f74115819b6a4c8548"
INSTALL_FILES = (
    "__init__.py",
    "launcher.py",
    "clients.py",
    "paths.py",
    "models.py",
    "catalog.py",
    "requirements.txt",
    "splash-engine",
)
COMPLETION_FILES = ("models", "_splash", "splash.bash", "official-models.txt")
SERVER_FILES = (
    "__init__.py",
    "server.py",
    "backend.py",
    "constraints.py",
    "output.py",
    "frontend.py",
    "judgments.py",
    "diagnostics.py",
    "api_shapes.py",
    "tool_schema.py",
    "tokenization.py",
    "json_codec.py",
    "latency.py",
    "metrics.py",
    "errors.py",
    "runtime.py",
    "protocol.py",
    "images.py",
    "documents.py",
    "document_worker.py",
    "http_security.py",
    "thinking.py",
    "schema_validation.py",
    "crash_trace.py",
    "chat.html",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_runtime(destination, version):
    for folder, names in (
        ("install", INSTALL_FILES),
        ("install/completions", COMPLETION_FILES),
        ("server", SERVER_FILES),
        ("engine", ("splash", "splash.metallib")),
    ):
        (destination / folder).mkdir()
        source = ROOT / ("build" if folder == "engine" else folder)
        for name in names:
            shutil.copy2(source / name, destination / folder / name)
    for name in ("LICENSE",):
        shutil.copy2(ROOT / name, destination / name)
    (destination / "release.json").write_text(
        json.dumps(
            {
                "version": version,
                "binary_sha256": digest(destination / "engine/splash"),
                "metallib_sha256": digest(destination / "engine/splash.metallib"),
            },
            indent=2,
        )
        + "\n"
    )


def formula(version, url, checksum, macos_min):
    # Inputs are encoded as Ruby double-quoted literals with the backslash,
    # the quote and the interpolation opener escaped, so no path or URL can
    # become executable Ruby; double quotes are what `brew audit` expects.
    def quote(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("#{", "\\#{")
        return '"' + escaped + '"'

    return f"""class SplashMacOSRequirement < Requirement
  fatal true
  satisfy(build_env: false) {{ OS.mac? && MacOS.full_version >= {quote(macos_min)} }}

  def message
    {quote(f"Splash requires macOS {macos_min} or newer.")}
  end
end

class Splash < Formula
  desc "Local inference engine for Apple silicon, built around the model"
  homepage "https://github.com/incoai/splash"
  url {quote(url)}
  sha256 {quote(checksum)}
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on macos: :tahoe
  depends_on SplashMacOSRequirement

  def install
    libexec.install Dir["*"]
    (bin/"splash").write <<~SH
      #!/bin/sh
      export PYTHONDONTWRITEBYTECODE=1
      exec "#{{opt_libexec}}/python/bin/python3" -u "#{{opt_libexec}}/install/launcher.py" "$@"
    SH
    chmod 0755, bin/"splash"
    zsh_completion.install_symlink libexec/"install/completions/_splash"
    bash_completion.install_symlink libexec/"install/completions/splash.bash" => "splash"
  end

  def caveats
    <<~CAVEAT
      Serve a model:
        splash serve --model incoai/Qwen3.8-27B-Splash
    CAVEAT
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/splash --version")
    assert_match "serve", shell_output("#{{bin}}/splash --help")
  end
end
"""


def _report_nonempty_staging(error, temporary_root):
    """Print the entries left in a staging directory after ENOTEMPTY."""
    failing_path = error.filename if error.filename is not None else temporary_root
    failing_directory = Path(os.fsdecode(failing_path))
    try:
        with os.scandir(failing_directory) as entries:
            remaining = sorted(entry.name for entry in entries)
        details = {
            "directory": str(failing_directory),
            "remaining_entries": remaining,
        }
    except OSError as diagnostic_error:
        details = {
            "directory": str(failing_directory),
            "diagnostic_error": str(diagnostic_error),
        }
    print(
        "package staging cleanup failed with ENOTEMPTY: "
        + json.dumps(details, sort_keys=True),
        file=sys.stderr,
    )


@contextlib.contextmanager
def _package_staging_directory():
    """Create private temporary staging and report, but never suppress, cleanup failures."""
    temporary = tempfile.TemporaryDirectory(prefix="splash-package-")
    try:
        yield Path(temporary.name)
    finally:
        try:
            temporary.cleanup()
        except OSError as error:
            if error.errno == errno.ENOTEMPTY:
                _report_nonempty_staging(error, temporary.name)
            raise


def _open_pending_archive(dist, archive_name):
    """Create a destination-side archive file with normal output permissions."""
    for _ in range(10):
        pending = dist / f".{archive_name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                pending,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666,
            )
        except FileExistsError:
            continue
        return pending, os.fdopen(descriptor, "wb")
    raise FileExistsError("could not allocate a unique pending package archive")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--macos-min", required=True, help="minimum macOS, e.g. 26.4")
    parser.add_argument(
        "--url", help="published tarball URL; defaults to GitHub Releases"
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error("invalid release version")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    name = f"splash-{args.version}-arm64-macos26"
    archive = dist / f"{name}.tar.gz"
    if archive.exists():
        parser.error(f"release already exists: {archive}")
    cached = ROOT / "build/release/python-runtime.tar.gz"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists() or digest(cached) != PYTHON_SHA256:
        with tempfile.NamedTemporaryFile(dir=cached.parent) as download:
            with urllib.request.urlopen(PYTHON_URL, timeout=60) as response:
                shutil.copyfileobj(response, download)
            download.flush()
            if digest(Path(download.name)) != PYTHON_SHA256:
                raise ValueError("Python distribution checksum mismatch")
            shutil.copyfile(download.name, cached)
    pending_archive = None
    try:
        with _package_staging_directory() as temporary:
            stage = temporary / name
            stage.mkdir()
            stage_runtime(stage, args.version)
            with tarfile.open(cached) as python:
                python.extractall(stage, filter="data")
            python = stage / "python/bin/python3"
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--only-binary=:all:",
                    "--no-compile",
                    "-r",
                    str(stage / "install/requirements.txt"),
                ],
                check=True,
            )
            # Exercise imports and the public entry point from the actual staged
            # runtime, not the developer's virtualenv or source PYTHONPATH.
            subprocess.run(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import sys; sys.path.insert(0, '.'); "
                    "import tokenizers, llguidance, numpy, PIL, transformers, huggingface_hub, pypdfium2; "
                    "import server.server, install.launcher",
                ],
                cwd=stage,
                check=True,
            )
            subprocess.run(
                [str(python), "-B", str(stage / "install/launcher.py"), "--help"],
                cwd=stage,
                check=True,
            )
            pending_archive, pending = _open_pending_archive(dist, archive.name)
            with pending:
                with tarfile.open(fileobj=pending, mode="w:gz") as release:
                    release.add(
                        stage,
                        arcname=name,
                        filter=lambda item: (
                            None if "__pycache__" in Path(item.name).parts else item
                        ),
                    )
                pending.flush()
                os.fsync(pending.fileno())
        pending_archive.replace(archive)
    except BaseException:
        if pending_archive is not None:
            try:
                pending_archive.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print(
                    f"could not remove incomplete package archive staging file "
                    f"{pending_archive}: {cleanup_error}",
                    file=sys.stderr,
                )
        raise
    checksum = digest(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(checksum + "\n")
    url = (
        args.url
        or f"https://github.com/incoai/splash/releases/download/{args.version}/{archive.name}"
    )
    (dist / "splash.rb").write_text(
        formula(args.version, url, checksum, args.macos_min)
    )
    print(
        f"Built {archive}; run make package-bottle RELEASE_VERSION={args.version} "
        "before publishing the Homebrew formula."
    )


if __name__ == "__main__":
    main()
