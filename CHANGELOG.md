# Changelog

## [Unreleased]

### Added

- Add paired GSM8K exact-answer benchmarks with direct HTTP and Pi harnesses.
- Allow the attention sweep to compare existing prefill attention configurations
  in rotating order, preserve individual timing samples, and report output
  differences against candidate tolerances.

### Changed

- Use a numerically qualified sequential Q4 decode baseline on 60-core Apple9
  GPUs and apply the measured Qwen3.8-27B M3 Ultra profile only to its exact
  package manifest.
- Apply the measured two-way Q8 prefill attention split only to the exact
  Qwen3.8-27B M3 Ultra package profile.
