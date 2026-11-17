# Fennec Fine-tune

Streamed LoRA fine-tuning of large MoE models (Qwen3.6-35B-A3B) on a small GPU,
built on **[AirLLM](https://github.com/lyogavin/airllm)**. The frozen bf16 base is
streamed one decoder layer at a time from disk shards; LoRA A/B + AdamW live on
CUDA and autograd hidden states stay on CPU. Verified on an 8 GB RTX 4060: a full
forward+backward through 40 MoE layers peaks at **~1.9 GB VRAM**.

## Memory budget

Measured during the 2026-11 audit with the since-retired preflight tooling:
**22.78 GiB weights** and **0.62 GiB of KV cache at 8,192 tokens** (81,920 bytes
per token), for a 23.40 GiB peak against 8.00 GiB VRAM + host RAM — weights tier
to the host pool, KV stays in memory. That tool no longer ships; treat the
figures as historical. The Fennec engine prints its own tiering decisions when
`serve` starts a model.

## Features

- Streamed LoRA training through AirLLM on an 8 GB GPU (~2 GB peak VRAM).
- Execution-verified SFT data generation: reference solutions are actually run against tests.
- Shard-wise LoRA merge, FQ4 sidecar quantization via the Fennec engine, and before/after evaluation.
- Frozen 60-case holdout with a per-case zero-regression promotion policy.
- Reproducible evidence: verdict JSON, run logs and comparison graphs in `results/`.

## Pipeline

1. **Download** the bf16 source — `scripts/download_qwen36_bf16.py`
   (`Qwen/Qwen3.6-35B-A3B` → `~/fennec-models`, cache in `~/.cache/fennec`).
2. **Build SFT data** — `scripts/make_sft_data.py` author coding tasks and
   *executes* each reference solution against unit tests, keeping only verified
   correct examples → `data/sft_coding_verified.jsonl`.
3. **Train** — `scripts/airllm_train_qwen36.py` (AirLLM `AirLLMLoRA`). Run
   `--smoke` first to confirm compatibility + VRAM, then a full run. Attention and
   shared-expert linears train; packed MoE experts stay frozen.
4. **Merge** — `scripts/merge_lora_qwen36.py` applies `W += (alpha/r)·(B@A)`
   shard-by-shard into a copy of the bf16 base.
5. **Convert + verify** — `scripts/run_finetune_eval.py` chains merge → convert
   (`fennec quant --layers all` writes an FQ4 sidecar) → evaluate → graph. The
   evaluate stage serves the baseline and the LoRA-merged candidate in turn on an
   owned port with the **Fennec engine**, runs the frozen 60-case holdout against
   each, and writes `results/finetune_verdict.json` from the suite's own
   `compare_reports` (quality-first: at least one gain and zero regressions; ties
   keep the incumbent). Retention-only scope: the verdict never promotes anything,
   and the original container is never modified. Requires the Fennec engine
   checkout (`~/Fennec` or `FENNEC_ENGINE_ROOT`) with a `--release` build.
6. **Graph** — `scripts/make_finetune_graph.py` reads `finetune_verdict.json`
   plus the training loss log and renders
   `results/bench_finetune_comparison.png`; the title states the retention
   verdict (regression, tie, or gains-only) and never claims promotion.

### Frozen holdout (60 cases)

`scripts/bench_holdout.py` is a stdlib-only, pinned benchmark across 12
structured-reasoning families (JSON merge patch, semver, CIDR, ISO calendar,
decimal rounding, ...) with exact JSON scoring. It never executes model output,
starts servers, or deletes models. Validate it offline with
`--manifest-only --out manifest.json`, run it against a serving engine with
`--model <id> --model-path <dir> --out report.json`, and compare two completed
reports with `--compare original.json candidate.json`. The comparison policy is
quality-first: a candidate is chosen only with at least one gain and zero
regressions; ties keep the incumbent.

```bash
FENNEC_API_KEY=local python scripts/bench_holdout.py \
  --model qwen36-35b --model-path ~/fennec-models/qwen36-35b \
  --out results/holdout_run_qwen36-35b.json
```

**Result (2026-11-10):** fine-tuned `qwen36-35b` scored **32/60**, base
`qwen36-35b-pre-finetune` scored **34/60**. The base gains 4 cases but
regresses 2 (`levenshtein.01`, `relational_division.03`), so the
zero-regression policy keeps the fine-tuned model. Evidence:
`results/holdout_run_qwen36-35b_finetuned.json`,
`results/holdout_run_qwen36-35b-pre.json`,
`results/holdout_compare_finetuned_vs_pre.txt`, `results/holdout_verdict.json`.
The suite was not used for training and remains a holdout.

![Holdout per-family accuracy](results/holdout_compare.png)

## Usage

```bash
python scripts/make_sft_data.py
python scripts/airllm_train_qwen36.py --smoke          # feasibility gate
python scripts/airllm_train_qwen36.py --epochs 2 --lr 5e-5 --rank 16
python scripts/merge_lora_qwen36.py --adapter ~/fennec-models/qwen36-lora-adapter.pt
python scripts/run_finetune_eval.py --stages merge convert evaluate graph
```

Stages are explicit opt-in (default `--stages evaluate`, which needs an already
merged `--merged` directory); each stage refuses to reuse an existing output.

## Credits

- **[AirLLM](https://github.com/lyogavin/airllm)** — layer-streamed inference and
  the streamed LoRA trainer this pipeline uses.

Part of the [Fennec suite](https://github.com/Liso7150/fennec-environment-setup).

## Engine vs AirLLM inference benchmark (historical)

`scripts/bench_engine_vs_airllm.py` compares an engine-backed inference path
with AirLLM-only inference on the same machine and prompts, and writes
`results/engine_vs_airllm.{json,md}`. The measurement below was recorded on
2026-11-11 against the then-serving Colibri engine; the suite now serves with
the Fennec engine, so the table stands as history, not as the current path.

```bash
python scripts/bench_engine_vs_airllm.py --backend both --tokens 32 --airllm-tokens 4
```

AirLLM streams every layer from disk for each token, so `--airllm-tokens` shrinks
its budget independently; a full 32-token sweep over three prompts takes hours.
Two gaps in stock AirLLM are closed so the comparison is a measurement rather than
a crash: checkpoints that nest the decoder under `model.language_model.layers` get
explicit layer names (stock AirLLM assumes `model.layers` and fails with
`ValueError: invalid literal for int() with base 10: 'layers'`), and a Qwen chat
template is supplied when the container ships none.

Result on 2026-11-11 (RTX 4060 8 GB, 22 threads, Python 3.12.10, `qwen36-35b`),
both phases driven by the same `python` on PATH:

| Backend | Load s | TTFT s | Decode tok/s | Peak RAM GiB | Peak VRAM GiB |
|---|---|---|---|---|---|
| Colibri engine (Vulkan) | 38.1 | 0.485 | **10.68** | 15.42 | 6.27 |
| AirLLM 4.0.0 only (CPU) | 2.4 | n/a | **0.077** | 1.37 | 0.36 |

**Verdict: an engine-backed path is the inference path — 139x faster decode
than AirLLM alone.** AirLLM's one advantage is footprint: 1.37 GiB RAM against 15.42 GiB, because it
holds a single layer at a time. That buys the memory with roughly 13 s per token,
which is acceptable for offline fine-tuning and unusable for chat. Decode rates
are compared like with like: the engine's excludes time-to-first-token, and
AirLLM's subtracts its timed 1-token warm-up to remove prefill. Full numbers and
per-prompt output: [`results/engine_vs_airllm.md`](results/engine_vs_airllm.md).

## Context-payload benchmark (old loop vs 1.4.1 compaction) — historical

Measured 2026-11-10 before the 1.0.0 engine switch; `fennec_core.context.compact()`
itself is current and still ships in fennec-core.

`scripts/bench_context_payload.py` replays one fixed agent transcript through an
already-running engine twice: the pre-1.4.1 payload (each tool result sliced to
6,000 characters, duplicates repeated verbatim) and `fennec_core.context.compact()`.
The arms alternate so warm-up drift lands on both; both are greedy with the same
`max_tokens`. It writes `results/context_payload.{json,md}`.

Measured 2026-11-10, `qwen3.6-colibri` (Vulkan, `--ctx 16384`, `max_tokens 8`,
2 pairs): **9,594 -> 6,995 prompt tokens, TTFT 70.70 s -> 51.14 s per step
(-19.56 s, -27.7 %)**. The script never starts or stops the engine, and it needs
only the standard library plus `fennec_core/context.py`, so it runs on any
Python 3.10+.

    python scripts/bench_context_payload.py --base-url http://127.0.0.1:8197 --exact-usage
