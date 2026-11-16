# Changelog

All notable changes to fennec-finetune, the AirLLM streamed LoRA pipeline and benchmark
harness.

The sections below carry the published tag each change actually shipped in
(`Liso7150/fennec-finetune`, `v0.3.0` … `v0.9.0`). An earlier revision of this
file numbered the same history on a 1.x/2.x ladder that was never published;
those tags still exist locally (`v1.5.0` … `v2.0.0`) and each one points at the
same commit as a published `v0.x` tag. The first release here was `0.3.0`, and
`0.7.0`, `0.7.1`, `0.8.0` and `0.9.0` all point at the same commit as `0.6.0` —
no new code shipped in them. Dates are the tag publish dates.

## [Unreleased] — 1.0.0 in preparation

Nothing at 1.0.0 is published. The newest release here is `v0.9.0`
(2026-11-11T19:29Z). Measured on the working tree: 105 tests passed plus 162
subtests (`python -m pytest -q`, 2026-11-15).

### Added
- README hero published, with the baselines it cites
- Every measured knob pinned under both the `FENNEC_*` and `COLI_*` spellings

### Changed
- Both spellings are stripped from the parent config, so the child wins
- The engine knobs are single-spelled: the harness pins `FENNEC_*` and emits the
  `COLI_*` names only when the resolved child is the legacy `c/coli` launcher
- The benchmark checkpoints moved to `D:/fennec-models` (was the MEGA-synced
  `colibri-models` directory)

## [0.9.0] - 2026-11-11

No new commit — the tag points at the same checkout as `0.6.0`.

## [0.8.0] - 2026-11-11

No new commit — the tag points at the same checkout as `0.6.0`.

## [0.7.1] - 2026-11-11

No new commit — the tag points at the same checkout as `0.6.0`.

## [0.7.0] - 2026-11-11

No new commit — the tag points at the same checkout as `0.6.0`.

## [0.6.0] - 2026-11-11

### Added
- The measured memory budget recorded

## [0.5.0] - 2026-11-11

### Added
- The committed results indexed
- AirLLM measured alone, fairly, against the engine

## [0.4.0] - 2026-11-11

### Added
- The low-RAM tier layout measured
- Tier profiles benchmarked: the GPU tier is worth 1.9× decode
- The compacted agent payload measured on the live engine
- The engine-vs-AirLLM benchmark

### Changed
- Model roots made drive-agnostic

## [0.3.0] - 2026-11-10

The first published release. It carries the pre-tag history below.

### Changed
- The fine-tune pipeline points at the Fennec model root

### Added
- A frozen 60-case holdout evaluation, its per-family comparison chart, safety
  tests and scratch cleanup
- An automated merge → convert → bench → graph evaluation chain

### Verdict
- Fine-tune promoted: 22/22 cases, zero regression

### Credits
- Initial commit as colibri-finetune — AirLLM streamed LoRA pipeline and
  benchmark harness (JustVugg/colibri + lyogavin/airllm)
