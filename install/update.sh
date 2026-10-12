#!/bin/sh
set -eu

source_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
python_executable=${SPLASH_INSTALL_PYTHON:-$(command -v python3)}
exec "${python_executable}" "${source_root}/install/splash_runtime.py" update \
    --source-root "${source_root}" "$@"
