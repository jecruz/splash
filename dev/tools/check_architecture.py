#!/usr/bin/env python3
"""Enforce production dependency boundaries."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".metal"}
INCLUDE = re.compile(r'^\s*#(?:include|import)\s+["<]([^">]+)[">]', re.MULTILINE)
OPERATOR_WORKSPACE_POLICY = re.compile(
    r"\b(?:PrefillAttentionWave|prefillAttentionTiles|"
    r"kQ8VerifySplits|kQ8PrefillAttentionTileRows|"
    r"moeMaximumTiles|kMoePrefillTileRows|kMoeDecodeTileRows|"
    r"Q4DecodeKind|Q4DecodeShape|Q4PrefillShape|kQ4PrefillTileRows|"
    r"narrowAffineKind|narrowResidualKind|headKind|gdnInputGroups|"
    r"attentionGroups|addPrefill128|LinearTile|LinearConfig|LinearSimdgroups|"
    r"PrefillSplitMultiplier|PrefillAttentionConfig|VerifySplitCount|"
    r"VerifyAttentionConfig|AttentionScalePlacement|MoeExpertTile|"
    r"MoeExpertSimdgroups|MoeConfig|"
    r"DraftAttentionConfiguration|selectorShards)\b"
)
QWEN38_PROFILE = re.compile(
    r"\bops::OperatorChoices\s+m3UltraQwen38Profile\s*\([^)]*\)\s*\{"
)
QWEN38_MANIFEST_SHA256 = (
    "40121d0d933fb27a7206248a592451f9ebbd73671f19ce3162cf27a7db0968d5"
)
QWEN38_PROFILE_GUARD = re.compile(
    r'\A\s*constexpr\s+std::string_view\s+manifestSha256\s*=\s*"'
    + QWEN38_MANIFEST_SHA256
    + r'";\s*if\s*\(\s*device\.appleGpuFamily\s*!=\s*9\s*\|\|\s*'
    + r"device\.gpuCoreCount\s*!=\s*60\s*\|\|\s*"
    + r'package\.name\(\)\s*!=\s*"Qwen3\.8-27B"\s*\|\|\s*'
    + r"digestHex\(package\.descriptor\.packageManifestSha256\)\s*!=\s*"
    + r"manifestSha256\s*\)\s*return\s*\{\s*\}\s*;",
    re.DOTALL,
)


def production_sources() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "runtime").rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def guarded_runtime_policy(name: str, text: str) -> tuple[str, str | None]:
    """Remove the one manifest-gated runtime profile before policy scanning."""
    if name != "runtime/engine/RuntimeResources.mm":
        return text, None

    matches = list(QWEN38_PROFILE.finditer(text))
    if not matches:
        return text, None
    if len(matches) != 1:
        return text, "runtime profile must have exactly one Qwen3.8 definition"

    match = matches[0]
    depth = 1
    end = match.end()
    while end < len(text) and depth:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    if depth:
        return text, "runtime Qwen3.8 profile has an incomplete body"

    body = text[match.end():end]
    if not QWEN38_PROFILE_GUARD.match(body):
        return text, "runtime Qwen3.8 profile is missing a required selector guard"
    return text[:match.start()] + text[end:], None


def check_server_dependencies() -> list[str]:
    # Request bridges and model-output handling must not import HTTP or prompt
    # preparation. The wire transport stays independent of those adapters.
    forbidden = {
        "backend": {"frontend", "api_shapes"},
        "constraints": {"frontend", "backend", "output", "api_shapes"},
        "output": {"frontend", "backend", "constraints", "api_shapes"},
        "runtime": {"frontend", "backend", "constraints", "output", "api_shapes"},
    }
    errors = []
    for path in sorted((ROOT / "server").glob("*.py")):
        if path.stem == "server":
            continue
        blocked = {"server", *forbidden.get(path.stem, ())}
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.module is None or (node.module == "server" and not node.level):
                    dependencies.update(alias.name for alias in node.names)
                else:
                    dependencies.add(node.module.removeprefix("server.").split(".")[0])
            elif isinstance(node, ast.Import):
                dependencies.update(
                    alias.name.removeprefix("server.").split(".")[0]
                    for alias in node.names
                )
        for dependency in sorted(dependencies & blocked):
            errors.append(f"{relative(path)}: imports upper serving layer {dependency}")
    return errors


def check() -> list[str]:
    errors = check_server_dependencies()
    obsolete_roots = (
        ROOT / "runtime" / "kernels",
        ROOT / "runtime" / "src" / "metal",
    )
    for path in obsolete_roots:
        if path.exists():
            errors.append(f"obsolete production directory exists: {relative(path)}")

    forbidden_metal_dependencies = ("engine/", "model/")
    forbidden_model_dependencies = ("engine/",)
    forbidden_operator_dependencies = ("engine/", "model/")
    engine_assembly_sources = {
        "runtime/engine/Bootstrap.hpp",
        "runtime/engine/Bootstrap.mm",
        "runtime/engine/RuntimeResources.hpp",
        "runtime/engine/RuntimeResources.mm",
    }
    concrete_model_headers = (
        "model/DFlashDraft.hpp",
        "model/ModelFactory.hpp",
        "model/Qwen3_6Moe.hpp",
        "model/Qwen3_8.hpp",
        "model/QwenState.hpp",
        "model/QwenTarget.hpp",
        "model/Runtime.hpp",
        "model/WeightStore.hpp",
    )
    concrete_model_symbols = re.compile(r"\bmodel::(?:Qwen\w*|DFlash\w*|Runtime)\b")
    client_names = re.compile(r"\b(?:Claude Code|OpenCode|Codex|Hermes)\b", re.I)

    for path in production_sources():
        name = relative(path)
        text = path.read_text(errors="replace")
        includes = INCLUDE.findall(text)
        for include in includes:
            if include.startswith("tuning/"):
                errors.append(f"{name}: production depends on offline tuning {include}")
        if name.startswith("runtime/metal/"):
            for include in includes:
                if any(part in include for part in forbidden_metal_dependencies):
                    errors.append(
                        f"{name}: Metal depends on production layer {include}"
                    )
        if name.startswith("runtime/model/"):
            for include in includes:
                if any(part in include for part in forbidden_model_dependencies):
                    errors.append(f"{name}: model depends on engine layer {include}")
            if re.search(r"\b(?:graph|commandGraph)\.add\s*\(", text):
                errors.append(f"{name}: model dispatches a Metal pipeline directly")
        if name.startswith(("runtime/model/", "runtime/engine/")):
            policy_text, profile_error = guarded_runtime_policy(name, text)
            if profile_error:
                errors.append(f"{name}: {profile_error}")
            elif OPERATOR_WORKSPACE_POLICY.search(policy_text):
                layer = name.split("/")[1]
                errors.append(f"{name}: {layer} owns an operator workspace policy")
        if name.startswith("runtime/ops/"):
            for include in includes:
                if any(part in include for part in forbidden_operator_dependencies):
                    errors.append(
                        f"{name}: operator depends on upper production layer {include}"
                    )
        if name.startswith("runtime/engine/") and name not in engine_assembly_sources:
            for include in includes:
                if include in concrete_model_headers:
                    errors.append(
                        f"{name}: engine policy depends on concrete model {include}"
                    )
        if name == "runtime/main.mm":
            for include in includes:
                if include in concrete_model_headers:
                    errors.append(
                        f"{name}: startup depends on concrete model {include}"
                    )
            if concrete_model_symbols.search(text):
                errors.append(f"{name}: startup names a concrete model type")
        if (
            name.startswith("runtime/engine/")
            and name != "runtime/engine/Checked.hpp"
            and re.search(r"^namespace splash\s*\{", text, re.MULTILINE)
        ):
            errors.append(f"{name}: engine declarations leak into root namespace")
        if client_names.search(text):
            errors.append(
                f"{name}: production backend contains client-specific behavior"
            )
    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"architecture error: {error}", file=sys.stderr)
        return 1
    print("backend architecture: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
