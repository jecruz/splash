"""Compare task-answer accuracy between matched Splash model binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dev.tests import smoke_real as smoke

DATASET_REPO = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATASET_FILE = "main/test-00000-of-00001.parquet"
DATASET_ROWS = 1319
ANSWER_PATTERN = re.compile(r"####\s*\$?\s*([-+]?[\d,]+(?:\.\d+)?)\s*$", re.MULTILINE)
PROFILE_LOG_LINE = "Installed validated M3 Ultra Qwen3.8-27B policy."
PREFILL_PROFILE_LOG_LINE = "Enabled two-way Q8 prefill attention profile."
PI_MODEL_IDS = {
    "incoai/Qwen3.8-27B-Splash": "incoai/Qwen3.8-27B-Splash",
    "incoai/Qwen3.6-35B-A3B-Splash": "incoai/Qwen3.6-35B-A3B-Splash",
}


def normalize_number(value: str) -> str:
    """Normalize an exact numeric answer without floating-point rounding."""
    try:
        number = Decimal(value.replace(",", "")).normalize()
    except InvalidOperation as error:
        raise ValueError(f"invalid numeric answer: {value!r}") from error
    if not number.is_finite():
        raise ValueError(f"non-finite numeric answer: {value!r}")
    return format(number, "f")


def extract_answer(output: str) -> str | None:
    """Return the last GSM8K-style final answer, or None if absent."""
    matches = ANSWER_PATTERN.findall(output)
    if not matches:
        return None
    try:
        return normalize_number(matches[-1])
    except ValueError:
        return None


def load_dataset(
    path: Path | None, limit: int, cache_dir: Path
) -> tuple[list[dict], dict]:
    """Load a fixed GSM8K sample from JSONL or the pinned Hub revision."""
    if path is not None:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        examples = []
        for row in rows:
            answer = row.get("expected", row.get("answer"))
            if answer is None:
                raise ValueError("JSONL dataset row has no expected answer")
            examples.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected": normalize_number(str(answer)),
                }
            )
        source = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    else:
        import duckdb
        from huggingface_hub import hf_hub_download

        parquet = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir),
        )
        parquet_sha = hashlib.sha256(Path(parquet).read_bytes()).hexdigest()
        rows = (
            duckdb.connect()
            .execute("SELECT question, answer FROM read_parquet(?)", [parquet])
            .fetchall()
        )
        if len(rows) != DATASET_ROWS:
            raise ValueError(
                f"pinned GSM8K test split has {len(rows)} rows, expected {DATASET_ROWS}"
            )
        examples = []
        for question, answer in rows:
            match = ANSWER_PATTERN.search(answer.strip())
            if match is None:
                raise ValueError("pinned GSM8K answer has no final-answer marker")
            key = hashlib.sha256(question.encode()).hexdigest()
            examples.append(
                {
                    "id": key[:16],
                    "question": question,
                    "expected": normalize_number(match.group(1)),
                    "sort_key": key,
                }
            )
        examples.sort(key=lambda row: row["sort_key"])
        for row in examples:
            del row["sort_key"]
        source = {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "file": DATASET_FILE,
            "file_sha256": parquet_sha,
            "sample": "lowest SHA-256(question) values",
        }
    examples.sort(key=lambda row: row["id"])
    if len(examples) < limit:
        raise ValueError(
            f"dataset has {len(examples)} examples, fewer than requested {limit}"
        )
    examples = examples[:limit]
    ids = [row["id"] for row in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset example IDs must be unique")
    source["selected_rows"] = len(examples)
    source["selected_ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return examples, source


def prompt_for(question: str) -> str:
    """Build the fixed user prompt used for both binaries."""
    return (
        "Solve this grade-school math problem. Output only one final line "
        "exactly in the form `#### <number>`.\n\n"
        f"Problem: {question}"
    )


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a binomial accuracy."""
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return center - half, center + half


def paired_bootstrap_interval(
    differences: list[int], *, seed: int = 1973
) -> tuple[float, float]:
    """Return a deterministic 95% percentile interval for paired accuracy delta."""
    if not differences:
        raise ValueError("paired interval requires at least one observation")
    generator = random.Random(seed)
    count = len(differences)
    draws = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(20_000)
    )
    return draws[499], draws[19_499]


def compare_quality(rows: list[dict]) -> dict:
    """Score exact answers and paired baseline-to-candidate changes."""
    versions = {}
    for version in ("baseline", "candidate"):
        selected = [row for row in rows if row["version"] == version]
        if not selected:
            raise ValueError(f"missing {version} quality outputs")
        correct = sum(row["correct"] for row in selected)
        low, high = wilson_interval(correct, len(selected))
        versions[version] = {
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected),
            "wilson_95": [low, high],
        }
    paired = {}
    for row in rows:
        paired.setdefault(row["id"], {})[row["version"]] = row["correct"]
    if any(set(pair) != {"baseline", "candidate"} for pair in paired.values()):
        raise ValueError(
            "quality examples must have paired baseline and candidate outputs"
        )
    differences = [
        int(pair["candidate"]) - int(pair["baseline"]) for pair in paired.values()
    ]
    low, high = paired_bootstrap_interval(differences)
    before_only = sum(value < 0 for value in differences)
    after_only = sum(value > 0 for value in differences)
    return {
        "baseline": versions["baseline"],
        "candidate": versions["candidate"],
        "paired": {
            "both_correct": sum(
                pair["baseline"] and pair["candidate"] for pair in paired.values()
            ),
            "both_incorrect": sum(
                not pair["baseline"] and not pair["candidate"]
                for pair in paired.values()
            ),
            "candidate_only_correct": after_only,
            "baseline_only_correct": before_only,
            "accuracy_delta": statistics.mean(differences),
            "bootstrap_95": [low, high],
        },
    }


def write_report(path: Path, document: dict) -> None:
    """Atomically persist a complete or partial quality report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_pi_agent_dir(agent_dir: Path, port: int, model: str) -> Path:
    """Preserve Pi settings and resources while adding an isolated Splash endpoint."""
    configured = Path.home() / ".pi" / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    settings = json.loads((configured / "settings.json").read_text())
    for resource in ("extensions", "skills", "prompts", "SYSTEM.md"):
        source = configured / resource
        if source.exists():
            (agent_dir / resource).symlink_to(
                source, target_is_directory=source.is_dir()
            )

    deterministic_extension = agent_dir / "splash-deterministic.js"
    deterministic_extension.write_text(
        "export default function (pi) {\n"
        "  pi.on('before_provider_request', (event) => ({\n"
        "    ...event.payload, temperature: 0, reasoning_effort: 'none',\n"
        "  }));\n"
        "}\n"
    )
    settings["extensions"] = [
        *settings.get("extensions", []),
        str(deterministic_extension),
    ]
    (agent_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (agent_dir / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "splash": {
                        "baseUrl": f"http://127.0.0.1:{port}/v1",
                        "api": "openai-completions",
                        "apiKey": "splash-local-benchmark",
                        "models": [
                            {
                                "id": PI_MODEL_IDS[model],
                                "name": model,
                                "contextWindow": 16384,
                                "maxTokens": 512,
                                "reasoning": False,
                                "input": ["text"],
                            }
                        ],
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )
    return deterministic_extension


def run_pi_prompt(
    prompt: str, model: str, agent_dir: Path, cwd: Path
) -> tuple[str, list[str]]:
    """Run Pi once and require a final assistant text result."""
    command = [
        "pi",
        "--mode",
        "json",
        "--no-session",
        "--thinking",
        "off",
        "--no-tools",
        "--system-prompt",
        "You solve math accurately. Output only one final line exactly in the "
        "form #### <number>.",
        "--provider",
        "splash",
        "--model",
        PI_MODEL_IDS[model],
        "--print",
        prompt,
    ]
    environment = os.environ.copy()
    environment.update(
        PI_CODING_AGENT_DIR=str(agent_dir),
        PI_OFFLINE="1",
        PI_SKIP_VERSION_CHECK="1",
        PI_TELEMETRY="0",
    )
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    events = []
    try:
        events = [
            json.loads(line) for line in result.stdout.splitlines() if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise smoke.SmokeFailure("Pi emitted a non-JSON event in JSON mode") from error
    messages = [
        event.get("message", {})
        for event in events
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
    ]
    text_messages = []
    for message in messages:
        text_messages.append(
            "".join(
                block.get("text", "")
                for block in message.get("content", [])
                if block.get("type") == "text"
            )
        )
    final_text = next((text for text in reversed(text_messages) if text.strip()), "")
    if result.returncode != 0 or not final_text:
        event_summary = [
            {
                "type": event.get("type"),
                "message_role": event.get("message", {}).get("role"),
                "stop_reason": event.get("message", {}).get("stopReason"),
                "content_types": [
                    block.get("type")
                    for block in event.get("message", {}).get("content", [])
                ],
            }
            for event in events[-8:]
        ]
        raise smoke.SmokeFailure(
            f"Pi returned no answer (exit={result.returncode}); "
            f"stderr={result.stderr[-1200:]!r}; "
            f"events={json.dumps(event_summary)}; stdout_tail={result.stdout[-1600:]!r}"
        )
    tool_names = [
        event.get("toolName", "unknown")
        for event in events
        if event.get("type") == "tool_execution_start"
    ]
    return final_text, tool_names


def parse_args(argv=None):
    """Parse and validate quality benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    smoke.add_server_arguments(parser)
    parser.add_argument("--baseline-binary", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="prepared JSONL sample; otherwise fetch the pinned GSM8K test set",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "splash-quality-hf-cache",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--harness", choices=("http", "pi"), default="http")
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument(
        "--output", type=Path, default=Path("build/release/quality-regression.json")
    )
    args = smoke.resolve_server_arguments(parser.parse_args(argv))
    if args.limit < 1 or args.max_completion_tokens < 1 or args.request_timeout <= 0:
        parser.error(
            "limit, max completion tokens, and request timeout must be positive"
        )
    if args.dataset is not None and not args.dataset.is_file():
        parser.error(f"dataset file does not exist: {args.dataset}")
    if args.binary == args.baseline_binary:
        parser.error("baseline and candidate binaries must be distinct")
    if args.harness == "pi" and args.model not in PI_MODEL_IDS:
        parser.error("Pi quality harness supports only the two qualified Splash models")
    return args


def run_version(
    version: str, binary: Path, args, examples: list[dict], pi_cwd: Path
) -> tuple[dict, list[dict]]:
    """Run the fixed quality set against one isolated Splash binary."""
    run_args = argparse.Namespace(**vars(args))
    run_args.binary = binary
    server = smoke.RealServer(run_args)
    pi_agent_dir = Path(tempfile.mkdtemp(prefix=f"splash-pi-{version}-"))
    results = []
    try:
        status = server.wait_ready(args.startup_timeout)
        smoke.validate_status(status, args.kv_format)
        startup_log = server.tail()
        is_qwen38 = args.model == "incoai/Qwen3.8-27B-Splash"
        expect_prefill_profile = version == "candidate" and is_qwen38
        if is_qwen38:
            smoke.require(
                PROFILE_LOG_LINE in startup_log,
                f"{version} Qwen3.8 M3 Ultra base profile was not installed",
            )
        elif PROFILE_LOG_LINE in startup_log:
            raise smoke.SmokeFailure("unexpected Qwen3.8 M3 Ultra base profile activation")
        if expect_prefill_profile:
            smoke.require(
                PREFILL_PROFILE_LOG_LINE in startup_log,
                "candidate Qwen3.8 two-way prefill profile was not installed",
            )
        elif PREFILL_PROFILE_LOG_LINE in startup_log:
            raise smoke.SmokeFailure("unexpected two-way prefill profile activation")
        if args.harness == "pi":
            prepare_pi_agent_dir(pi_agent_dir, server.port, args.model)
        for example in examples:
            prompt = prompt_for(example["question"])
            tool_names = []
            if args.harness == "pi":
                completion, tool_names = run_pi_prompt(
                    prompt, args.model, pi_agent_dir, pi_cwd
                )
            else:
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": args.max_completion_tokens,
                    "temperature": 0,
                    "reasoning_effort": "none",
                }
                code, response = smoke.request(
                    server.port,
                    "POST",
                    "/v1/chat/completions",
                    body,
                    timeout=args.request_timeout,
                )
                smoke.require(
                    code == 200,
                    f"quality request {example['id']} returned HTTP {code}: {response!r}",
                )
                completion = (
                    response.get("choices", [{}])[0].get("message", {}).get("content")
                    or ""
                )
            parsed = extract_answer(completion)
            results.append(
                {
                    "version": version,
                    "id": example["id"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "expected": example["expected"],
                    "parsed_answer": parsed,
                    "correct": parsed == example["expected"],
                    "output_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                    "output": completion,
                    "tool_calls": tool_names,
                }
            )
        return {
            "version": version,
            "binary": str(binary.resolve()),
            "identity": status["identity"],
            "profile_installed": PROFILE_LOG_LINE in startup_log,
            "prefill_profile_installed": PREFILL_PROFILE_LOG_LINE in startup_log,
        }, results
    except Exception as error:
        raise smoke.SmokeFailure(
            f"{version} quality benchmark failed: {error}\n{server.tail()}"
        ) from None
    finally:
        server.close()
        shutil.rmtree(pi_agent_dir, ignore_errors=True)


def main(argv=None) -> int:
    """Run and save the before/after task-quality comparison."""
    args = parse_args(argv)
    examples, dataset = load_dataset(args.dataset, args.limit, args.cache_dir)
    document = {
        "schema_version": 1,
        "benchmark": "GSM8K exact final-answer accuracy",
        "harness": args.harness,
        "dataset": dataset,
        "dataset_rows": len(examples),
        "prompt_policy": "fixed exact-answer prompt, temperature 0, reasoning disabled",
        "models": [],
        "samples": [],
        "benchmark_completed": False,
        "command": sys.argv,
        "python": platform.python_version(),
    }
    pi_cwd = Path(tempfile.mkdtemp(prefix="splash-pi-workspace-"))
    try:
        for version, binary in (
            ("baseline", args.baseline_binary),
            ("candidate", args.binary),
        ):
            model, results = run_version(version, binary, args, examples, pi_cwd)
            document["models"].append(model)
            document["samples"].extend(results)
        document["comparison"] = compare_quality(document["samples"])
        document["benchmark_completed"] = True
    finally:
        write_report(args.output, document)
        shutil.rmtree(pi_cwd, ignore_errors=True)
    print(json.dumps(document["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
