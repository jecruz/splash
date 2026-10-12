# Splash

[![CI](https://github.com/incoai/splash/actions/workflows/ci.yml/badge.svg)](https://github.com/incoai/splash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-black.svg)](#quick-start)

**A local inference engine for Apple silicon, built around the model.**

Splash serves a small set of models to coding agents and to any OpenAI or
Anthropic compatible client, on one Mac. On a 48 GB M5 Pro it decodes
Qwen3.8-27B at 2× the speed of the next-fastest engine and, with a 32K context
cached, returns the first token in 282 ms. Its kernels, draft model, and memory
plan are specialized for each model it serves. That is why it is fast, and why
there is nothing to configure.

## Quick start

Apple M3 or newer, macOS 26.4 or later, [Homebrew](https://brew.sh), and 36 GB
of unified memory (48 GB or more recommended).

```bash
brew install incoai/tap/splash
splash serve --model incoai/Qwen3.8-27B-Splash
```

The first run downloads and verifies the model package, checks available
memory, and starts serving on `127.0.0.1:8318`.

Once it prints `Ready`, leave this terminal open. Open <http://127.0.0.1:8318>
in your browser, or run an installed coding agent from another terminal:

```bash
splash opencode    # or: splash claude / splash codex / splash hermes
```

Press Ctrl+C in the server terminal to stop Splash.

## Use the API

Splash speaks OpenAI Chat Completions (`/v1/chat/completions`), OpenAI Responses
(`/v1/responses`), and Anthropic Messages (`/v1/messages`), all with streaming,
tool calls, JSON Schema output, images, and inline PDFs. `/tokenize` and
`/apply-template` return token IDs and the rendered prompt without running the
model.

```bash
curl http://127.0.0.1:8318/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "incoai/Qwen3.8-27B-Splash",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
```

`model` is optional. If set, use the served package ID or a configured
[model alias](DEVELOPMENT.md#api-model-aliases).
Reasoning follows the model default unless a [server default](DEVELOPMENT.md#default-reasoning-effort)
is configured. `"reasoning_effort": "none"` turns it off, and
Qwen3.8-27B also takes `low`, `medium`, and `xhigh`.

`/v1/judgments` and `/v1/systemone` provide scoring without generation.
See [judgment contracts](DEVELOPMENT.md#judgment-contracts) for details.

## Models

| Package (`--model`) | Contents | Download |
| --- | --- | ---: |
| [`incoai/Qwen3.8-27B-Splash`](https://huggingface.co/incoai/Qwen3.8-27B-Splash) | Qwen3.8-27B, 4-bit, with its DFlash 2 draft | 17.4 GB |
| [`incoai/Qwen3.6-35B-A3B-Splash`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-Splash) | Qwen3.6-35B-A3B, 4-bit, with its DFlash 2 draft | 20.9 GB |

`--model` takes any `owner/repo` that holds a Splash package, a format
[DEVELOPMENT.md](DEVELOPMENT.md#model-packages) describes. Plain MLX or
Transformers checkpoints do not work. Private repositories need `HF_TOKEN`.
Packages download into the Hugging Face cache, and `brew upgrade splash` keeps
them, along with model links and agent sessions.

For LM Studio Bionic, follow its [Splash setup guide](https://lmstudio.ai/blog/splash-engine):
install the Splash runtime, then paste the full Hugging Face model link into its
model search. These integrations manage their own runtime and settings.

If a client’s model catalog does not list a package, the full repository ID in
this table still works with `splash serve --model OWNER/REPO`. The browser chat
and the agent launchers connect to that server without a catalog search.

For a custom model download location, see [model cache](DEVELOPMENT.md#model-cache).

## Settings

There is no config file. The server binds `127.0.0.1:8318` by default.
Context supports up to the model’s native 256K window; usable capacity
depends on available memory. `splash serve --help` lists server options and examples.
The startup summary and `maximum_context_tokens` in `/status` show the effective
server limit. `/v1/models` and `/v1/models/{id}` report the same limit as
`max_model_len` and its compatibility alias `context_length`, including model aliases.
Clients can impose a smaller limit. With enough memory,
request the full window using `--max-context 256K`. This is a capacity limit, not a guarantee
that a long uncached prompt will reach its first token quickly.

`splash serve` accepts these optional flags:

- `--host`: HTTP bind address. Default: `127.0.0.1`.
- `--port`: HTTP port. Defaults to `SPLASH_PORT` or `8318`.
- `--max-memory`: ceiling on Metal allocations, e.g. `28G`. Default: auto.
- `--max-context`: context limit, up to `256K`, e.g. `100K`. Default: auto.
- `--max-prefix-cache`: logical idle reusable KV plus recurrent-state cache
  budget, e.g. `8G`. The same 8 GiB default applies to every model; `0`
  disables retention after requests complete.
- `--kv-format`: target KV cache storage, `int8` (default) or `bf16`.
- `--max-image-pixels`: maximum resized pixels per image. Default: 4,194,304.
- `--allowed-host`: extra HTTP `Host` name to accept, not a bind address. Repeatable.
- `--api-key`: require this key on API requests, as a bearer token or
  `x-api-key`. Defaults to `SPLASH_API_KEY`.
- `--no-webui`: turn off the chat page.

## Local fork install

The `jecruz/splash` fork has a user-local release manager separate from the
Homebrew package. Run these commands from a clean checkout whose `HEAD` equals
the fetched `origin/main`; the qualification receipt is tied to that exact
commit. GitHub Actions is not enabled on the fork, so qualification records the
local CI-equivalent targets and reports Woodpecker as not configured until the
private mirror is available.

```sh
./install/qualify.sh
./install/install.sh --qualification "$HOME/.local/state/splash/qualifications/<commit>/<attempt>/qualification.json"
~/.local/bin/splash-engine serve --model incoai/Qwen3.8-27B-Splash
./install/verify.sh
./install/update.sh --qualification "$HOME/.local/state/splash/qualifications/<commit>/<attempt>/qualification.json"
./install/rollback.sh
./install/uninstall.sh
```

Qualification runs `make check-source`, `make check-native-cpu`,
`make test-sanitizers`, and `make check-python-engine` in fresh Python 3.12,
3.13, and 3.14 environments. Native builds, sanitizers, Python gates, and
package builds run through `~/.local/bin/heavy-op` with a 16 GiB reservation.
The installer validates the archive checksum and embedded `release.json`
hashes before creating the immutable
`~/.local/share/splash/releases/<full-commit-sha>` directory. It atomically
updates `current` and retains `previous`; the manifest is available at
`~/.local/share/splash/current/manifest.json`. The stable launcher uses only
the selected release and defaults to `127.0.0.1:8318` (`SPLASH_PORT` or
`--port` can override it), leaving Woodpecker's port 8000 untouched.

`./install/verify.sh` checks the current links, qualification receipt,
manifest, launcher version, engine and metallib hashes. An optional
`--smoke-model OWNER/REPO --smoke-need-gb N` starts a model only if its package
is already cached, uses a temporary port, and keeps that server lifetime under
`heavy-op`. The launcher passes `SPLASH_API_KEY` through its inherited process
environment; the key is never placed in command-line arguments, so a provider
can supply it from Keychain. `splash-engine` rejects `--api-key`; configure
`SPLASH_API_KEY` in the launching process environment instead. Rollback swaps
the validated `current` and `previous` releases. Point a client at
`http://127.0.0.1:8318/v1` and pass
`SPLASH_API_KEY` through its environment when authentication is enabled.
Uninstall removes only this installer's managed release root and
`splash-engine`; it does not remove the legacy `splash-1974` shim or any
`~/.local/opt/splash-*` directory.

To use BF16 target KV, select it when starting the server:

```bash
splash serve --model incoai/Qwen3.8-27B-Splash --kv-format bf16
```

BF16 avoids target KV quantization, uses approximately twice the target KV
memory, and can be slower at long contexts. Model weights are unchanged.
Restart the server to switch formats. Omit `--kv-format` or use
`--kv-format int8` for the default INT8 cache.

If the model does not fit in the memory available, startup prints a memory
budget breakdown and stops.

The prefix-cache limit is enforced after each request completes. When retained
KV and recurrent-state entries exceed the limit, Splash evicts disposable
checkpoints first, then the least-recently-used eligible state or state-free KV
leaf. Exact token-page and image identities are unchanged. Active requests and
pinned recurrent states are never evicted, so their bytes can temporarily put
the logical cache above the configured limit; `/status` reports that overage.
The limit measures logical reusable cache ownership, not Metal allocator
backing or process memory. KV pool backing, recurrent-state residency, and Metal
current/peak allocation are reported separately because eviction may make
backing reusable without immediately returning it to the operating system.
The HTTP `/status` response also reports the frontend process's current resident
size and physical footprint under `process_memory`; on macOS these values come
from `proc_pid_rusage_info_v2` and are separate from the native engine's Metal
allocation counters.

Authentication is off by default. Set `SPLASH_API_KEY` in the shell that runs
`splash serve` and in the shell that runs an agent, and both sides use it.
Health and readiness probes stay public.

For LAN access and multiple servers, see
[server configuration](DEVELOPMENT.md#server-configuration).

Experimental cache offloading: [PR #3](https://github.com/incoai/splash/pull/3).

## Performance

Measured on an M5 Pro (16-core GPU, 48 GB): selected SPEED-Bench coding prompts
over HTTP, a 1,024-token output limit, reasoning on (medium for the 27B). The
ratio in each cell is against the next-fastest engine we measured.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 210 tok/s (1.7×) | 74 tok/s (2.0×) |
| Prefill · 32K prompt | 2,011 tok/s (1.3×) | 363 tok/s (1.2×) |
| Cached time to first token · 32K replay | 123 ms (6.6×) | 282 ms (7.3×) |
| Aggregate decode · 4 concurrent short prompts | 357 tok/s (2.0×) | 170 tok/s (3.9×) |

Splash led on every measure at every prompt length we tested, and the lead
grows with load: 3.8× at four concurrent 32K requests on the 35B. The
[launch post](https://inco.ai/blog/splash/) has the method and the full
comparison against oMLX, Lily, uzu, and Ollama.

For repeatable measurements on your Mac, see [local benchmarks](DEVELOPMENT.md#local-benchmarks).

## Design

The runtime, scheduler, cache, and API are shared. Everything else is rebuilt
per model:

- **A draft trained for the model.** Speculative decoding is the decode path in
  Splash, not an option. Every model ships with its own [DFlash
  2](https://inco.ai/blog/dflash2/) draft, and one pass of the target verifies a
  block of tokens in parallel.
- **Kernels compiled for exact shapes.** Fused Metal kernels, written and tuned
  by our in-house kernel agents for the model's dimensions, read weights packed
  for them and mapped zero-copy from disk. They ship precompiled: no Xcode, no
  compiler toolchain, nothing tuned on your machine.
- **A memory plan computed for this machine.** Context, KV capacity, and batch
  limits are worked out at startup from the memory Metal recommends, less the
  weights, the draft, and each request's state.

The [launch post](https://inco.ai/blog/splash/) covers the design in depth.

## More

- [DEVELOPMENT.md](DEVELOPMENT.md): building from source, tests, model packages,
  and release packaging.
- Apache-2.0, see [LICENSE](LICENSE). Model weights keep their own licenses.
