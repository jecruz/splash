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
RAW_STRING_PREFIX = re.compile(r"(?:u8|u|U|L)?R\"")


def _skip_quoted_literal(text: str, start: int, quote: str) -> int | None:
    """Return the first index after a quoted C++ literal, or None if incomplete."""
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            if text.startswith("\\\r\n", index):
                index += 3
            else:
                index += 2
        elif text[index] == quote:
            return index + 1
        else:
            index += 1
    return None


def _skip_raw_string(text: str, start: int) -> int | None:
    """Return the first index after a raw C++ string, or None if start is not one."""
    if start and (text[start - 1].isalnum() or text[start - 1] == "_"):
        return None
    prefix = RAW_STRING_PREFIX.match(text, start)
    if prefix is None:
        return None
    opening = text.find("(", prefix.end())
    if opening < 0:
        return None
    delimiter = text[prefix.end() : opening]
    if len(delimiter) > 16 or any(char.isspace() or char in "()\\" for char in delimiter):
        return None
    closing = text.find(")" + delimiter + '"', opening + 1)
    if closing < 0:
        return None
    return closing + len(delimiter) + 2


def _is_numeric_digit_separator(text: str, index: int) -> bool:
    """Recognize digit separators so they are not parsed as character literals."""
    if index == 0 or index + 1 >= len(text):
        return False
    previous, following = text[index - 1], text[index + 1]
    if previous.isdigit() and following.isdigit():
        return True
    if previous.lower() not in "abcdef" or following.lower() not in "abcdef0123456789":
        return False
    token_start = index - 1
    while token_start > 0 and (text[token_start - 1].isalnum() or text[token_start - 1] == "_"):
        token_start -= 1
    return text[token_start : token_start + 2].lower() == "0x"


def _after_line_splices(text: str, index: int) -> int:
    """Skip C++ backslash-newline pairs beginning at index."""
    while text.startswith("\\\r\n", index) or text.startswith("\\\n", index):
        index += 3 if text.startswith("\\\r\n", index) else 2
    return index


def _skip_cpp_non_code(text: str, index: int) -> int | None:
    """Return the end of a C++ comment or literal starting at index."""
    after_slash = _after_line_splices(text, index + 1) if text[index] == "/" else index
    if text[index] == "/" and after_slash < len(text) and text[after_slash] == "/":
        index = after_slash + 1
        while index < len(text):
            spliced = _after_line_splices(text, index)
            if spliced != index:
                index = spliced
                continue
            if text.startswith("\r\n", index):
                return index + 2
            if text[index] in "\r\n":
                return index + 1
            index += 1
        return index
    if text[index] == "/" and after_slash < len(text) and text[after_slash] == "*":
        index = after_slash + 1
        while index < len(text):
            if text[index] == "*":
                after_star = _after_line_splices(text, index + 1)
                if after_star < len(text) and text[after_star] == "/":
                    return after_star + 1
            index += 1
        return None
    raw_end = _skip_raw_string(text, index)
    if raw_end is not None:
        return raw_end
    if text[index] == "'" and _is_numeric_digit_separator(text, index):
        return None
    if text[index] in ('"', "'"):
        return _skip_quoted_literal(text, index, text[index])
    return None


def _mask_cpp_non_code(text: str, mask_literals: bool = True) -> str:
    """Blank C++ comments and optionally literals while preserving offsets."""
    masked = list(text)
    index = 0
    while index < len(text):
        end = _skip_cpp_non_code(text, index)
        if end is None:
            index += 1
            continue
        is_literal = text[index] in ('"', "'") or _skip_raw_string(text, index) is not None
        if not is_literal or mask_literals:
            for offset in range(index, end):
                if masked[offset] not in "\r\n":
                    masked[offset] = " "
        index = end
    return "".join(masked)


def _mask_inactive_if_zero(text: str) -> str:
    """Blank literal #if 0 branches so decoy profiles cannot be selected."""
    masked = list(text)
    stack: list[tuple[bool, bool, bool]] = []
    inactive = False
    line_start = 0
    for line in text.splitlines(keepends=True):
        content_end = line_start + len(line.rstrip("\r\n"))
        directive = re.match(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line[: content_end - line_start])
        if directive:
            kind, condition = directive.groups()
            if kind == "if":
                zero_branch = condition.strip() == "0"
                stack.append((inactive, zero_branch, False))
                inactive = inactive or zero_branch
            elif kind in {"ifdef", "ifndef"}:
                stack.append((inactive, False, False))
            elif kind == "else" and stack:
                parent_inactive, zero_branch, _ = stack[-1]
                stack[-1] = (parent_inactive, zero_branch, True)
                inactive = parent_inactive
            elif kind == "elif" and stack:
                parent_inactive, _, _ = stack[-1]
                stack[-1] = (parent_inactive, False, False)
                inactive = parent_inactive
            elif kind == "endif" and stack:
                parent_inactive, _, _ = stack.pop()
                inactive = parent_inactive
        elif inactive:
            for offset in range(line_start, line_start + len(line)):
                if masked[offset] not in "\r\n":
                    masked[offset] = " "
        line_start += len(line)
    return "".join(masked)


def _matching_brace_end(text: str, opening: int) -> int | None:
    """Find a matching C++ brace while ignoring comments and string literals."""
    depth = 1
    index = opening + 1
    while index < len(text):
        end = _skip_cpp_non_code(text, index)
        if end is not None:
            index = end
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


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

    signature_view = _mask_inactive_if_zero(_mask_cpp_non_code(text))
    matches = list(QWEN38_PROFILE.finditer(signature_view))
    if not matches:
        return text, None
    if len(matches) != 1:
        return text, "runtime profile must have exactly one Qwen3.8 definition"

    match = matches[0]
    end = _matching_brace_end(text, match.end() - 1)
    if end is None:
        return text, "runtime Qwen3.8 profile has an incomplete body"

    body = _mask_cpp_non_code(text[match.end() : end], mask_literals=False)
    body = _mask_inactive_if_zero(body)
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
