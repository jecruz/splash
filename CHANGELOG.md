# Changelog

## [Unreleased]

### Added

- Add the fork-local `splash-engine` launcher and qualified, versioned install,
  verify, update, rollback, and uninstall lifecycle.
- Bound retained native KV and recurrent-state prefixes with a shared default
  budget, deterministic LRU eviction, and `/status` accounting.
- Expose `--max-prefix-cache` on `splash serve` to set or disable that budget.
- Report the HTTP frontend process's resident size and physical footprint in
  `/status` for cache-memory attribution.
- Add paired GSM8K exact-answer benchmarks with direct HTTP and Pi harnesses.
- Allow the attention sweep to compare existing prefill attention configurations
  in rotating order, preserve individual timing samples, and report output
  differences against candidate tolerances.

### Changed

- Use port 8318 as the default local listener to avoid the fork's Woodpecker port.
- Use a numerically qualified sequential Q4 decode baseline on 60-core Apple9
  GPUs and apply the measured Qwen3.8-27B M3 Ultra profile only to its exact
  package manifest.
- Apply the measured two-way Q8 prefill attention split only to the exact
  Qwen3.8-27B M3 Ultra package profile.

### Fixed

- Resolve `heavy-op` from the account database home so local qualification works
  when CI gives the build an isolated `HOME`.
- Bind installable package bytes to the local qualification receipt and build them
  from a fresh detached worktree of the qualified fork commit.
- Reject abbreviated API-key options so launcher credentials stay environment-only.
- Allow the architecture check's manifest-gated Qwen3.8 assembly profile while
  continuing to reject operator workspace policy elsewhere.
- Stage release package trees in the system temporary directory and report
  remaining entries when staging cleanup fails with `ENOTEMPTY`.
- Give the real-pipe partial-write regression test a wider I/O timeout and
  matching request/result bounds so CI scheduling cannot race its status check.
