![Fennec Fine-tune 1.0.0 — streamed LoRA fine-tuning of large MoE models (Qwen3.6-35B-A3B) on a small GPU, built on AirLLM, plus the benchmark harnesses that decide whether a candidate is worth keeping. Python 3.10+, CUDA 8 GB, LoRA not QLoRA, 105 tests, 60 holdout cases, MIT license.](docs/hero.png)

## At a glance

| Tests | Test files | Holdout cases | Families | Pipeline scripts | int4 weights |
|---|---|---|---|---|---|
| 105 | 4 | 60 | 12 | 11 | 22.777 GiB |

Sources: counted in `tests/` and `scripts/`; `scripts/bench_holdout.py` (12 families × 5
cases); `results/preflight_qwen36_35b_ctx8192.json`.

The frozen bf16 base is streamed one decoder layer at a time from disk shards; LoRA A/B and
AdamW live on CUDA, and autograd hidden states stay on CPU. This is **LoRA, not QLoRA and not
a full fine-tune**: attention and shared-expert linears train, packed MoE experts stay frozen.

## Pipeline

```
bf16 download ─► verified SFT data ─► AirLLM LoRA train ─► shard-wise merge ─► bf16 + LoRA copy
   ├─ quant + verify: FQ4 sidecar ─► baseline and candidate serve ─► 60 holdout cases each
   └─ graph: passes and training loss ─► results/bench_finetune_comparison.png
```

| Step | Script | Notes |
|---|---|---|
| Download | `scripts/download_qwen36_bf16.py` | `Qwen/Qwen3.6-35B-A3B` → `~/fennec-models`, cache in `~/.cache/fennec` |
| SFT data | `scripts/make_sft_data.py` | executes each reference solution against unit tests, keeps only verified-correct examples; repo copy: `data/sft_coding_verified.jsonl`; `FENNEC_RESULTS` overrides the results root |
| Train | `scripts/airllm_train_qwen36.py` | AirLLM `AirLLMLoRA`; run `--smoke` first to confirm compatibility and VRAM on your card. Defaults: rank 16, alpha 32, lr 5e-5, seq 768, 2 epochs, at most 40 steps |
| Merge | `scripts/merge_lora_qwen36.py` | `W += (alpha/r)·(B@A)` shard-by-shard into a copy of the bf16 base; `weights_only=True` |
| Quant + verify | `scripts/run_finetune_eval.py` | merge → `fennec quant --layers all` (the FQ4 sidecar lands in the run directory, never in the model tree) → baseline and candidate served one at a time and scored by `bench_holdout.py` → `verdict()` → `results/<run-id>/finetune_verdict.json` → graph |
| Graph | `scripts/make_finetune_graph.py` | passes and training loss → `results/bench_finetune_comparison.png` |

> [!IMPORTANT]
> Nothing promotes automatically. Every run plan records `scope: retention_only` and
> `promotion_eligible: false`, and `verdict()` only reports what
> `bench_holdout.compare_reports()` measured: the candidate is *recommended* on at least one gain
> and zero regressions, a tie keeps the incumbent, and no model is ever moved or deleted. The
> archived `results/finetune_verdict.json` says "promote candidate (no regression)"; that verdict
> was written by an earlier version of the script and is kept as a record of its run, not as
> current policy.

## Memory budget

Measured with `preflight.plan(model_dir, ctx=8192)` against the production int4
container. The preflight reads `config.json` only — it loads no weights:

| Quantity | Value |
|---|---|
| Weights | **22.777 GiB** (24,456,460,083 B) |
| KV per token | 81,920 B |
| KV at 8,192 tokens | **0.625 GiB** (671,088,640 B) |
| Peak | **23.402 GiB** (25,127,548,723 B) |
| VRAM on the measured host | 7.996 GiB |
| Verdict | `weights_fit: hybrid`, `kv_fits: true` |

Weights tier to the host pool; KV stays in memory. Evidence:
[`results/preflight_qwen36_35b_ctx8192.json`](results/preflight_qwen36_35b_ctx8192.json).
Reproduce with `fennec preflight --model <model-dir> --ctx 8192 --json`.

## Engine: Fennec only

Every script that starts an engine resolves it through
[`scripts/engine_paths.py`](scripts/engine_paths.py), the same convention `fennec-core`
uses: probe `target/release`, `target/debug`, then the engine root for `fennec.exe`/`fennec`.
Fennec is native Rust and self-launches — no Python prefix — and it is the **only** engine;
the upstream Colibri launcher is retired and is never resolved or started:

```bash
cd ~/Fennec && cargo build --release
./target/release/fennec serve --auto-tier --model ~/fennec-models/qwen36-35b \
  --host 127.0.0.1 --port 8184
```

`--engine-root` (or `FENNEC_ENGINE_ROOT` / `FENNEC_HOME`) selects the checkout; the default is
`~/Fennec`. The engine speaks the OpenAI-compatible HTTP contract on port 8184 and reads one
spelling of each knob, `FENNEC_*` (`FENNEC_MODEL`, `FENNEC_API_KEY`, `FENNEC_VULKAN`,
`FENNEC_METAL`, `FENNEC_CUDA`, `FENNEC_VK_TIER_HOST_GB`). Nothing is written under the retired
`COLI_*` names; stale `COLI_*` shell exports are scrubbed before a measurement run so they
cannot influence it.

| Script | Engine |
|---|---|
| `bench_holdout.py`, `bench_context_payload.py` | Fennec (HTTP only, engine never started) |
| `bench_engine_vs_airllm.py`, `bench_tier_profiles.py` | Fennec (starts `fennec serve`) |
| `run_finetune_eval.py` | Fennec (`target/release/fennec` directly; `quant` and `--fq4` are Fennec commands) |

`run_finetune_eval.py` needs a Fennec checkout under `--engine-root` because its quant and serve
stages are Fennec's own commands. Models without a Fennec build are evaluated by pointing
`bench_holdout.py` at any OpenAI-compatible endpoint instead.

## Usage

```bash
python scripts/make_sft_data.py
python scripts/airllm_train_qwen36.py --smoke          # feasibility gate
python scripts/airllm_train_qwen36.py --epochs 2 --lr 5e-5 --rank 16
python scripts/merge_lora_qwen36.py --adapter ~/fennec-models/qwen36-lora-adapter.pt
python scripts/run_finetune_eval.py --stages merge convert evaluate graph
FENNEC_API_KEY=local python scripts/bench_holdout.py --model qwen36-35b \
  --model-path ~/fennec-models/qwen36-35b --out results/holdout_fennec.json
```

## Frozen holdout (60 cases)

`scripts/bench_holdout.py` is a stdlib-only, pinned benchmark across 12 structured-reasoning families
(JSON merge patch, semver, CIDR, ISO calendar, decimal rounding, ...) with exact JSON scoring. It never
executes model output, starts servers, or deletes models. Policy is quality-first: a candidate is chosen
only with at least one gain and zero regressions; ties keep the incumbent.

```bash
python scripts/bench_holdout.py --manifest-only --out manifest.json   # offline validation
FENNEC_API_KEY=local python scripts/bench_holdout.py \
  --model qwen36-35b --model-path ~/fennec-models/qwen36-35b \
  --out results/holdout_run_qwen36-35b.json
python scripts/bench_holdout.py --compare original.json candidate.json
```

Defaults: `--base http://127.0.0.1:8184/v1`, `--max-tokens 384`, `--timeout 120`.

**Recorded result (2026-11-10):**

| Model | Passed | Gains | Regressions | Ties |
|---|---|---|---|---|
| `qwen36-35b` (fine-tuned, incumbent) | 32/60 | — | — | — |
| `qwen36-35b-pre-finetune` (base, candidate) | 34/60 | 4 | 2 (`levenshtein.01`, `relational_division.03`) | 54 |

The zero-regression policy keeps the fine-tuned model: `recommendation: keep_original`. Evidence:
`results/holdout_verdict.json`, `results/holdout_run_qwen36-35b_finetuned.json`,
`results/holdout_run_qwen36-35b-pre.json`, `results/holdout_compare_finetuned_vs_pre.txt`. The suite was
not used for training; scores are not a statistical-significance claim and do not measure sandboxed execution.

> [!NOTE]
> The archived reports carry the suite label `colibri-holdout-1.0.0`. `bench_holdout.py` now
> emits `fennec-holdout-1.0.0` with an unchanged case fingerprint (SHA-256
> `4a627cfc…dcc85`), so `--compare` refuses those archives until they are re-run. The
> recorded verdicts stand as evidence of the runs that produced them.

![Holdout per-family accuracy](results/holdout_compare.png)

## Engine vs AirLLM inference benchmark

`scripts/bench_engine_vs_airllm.py` compares the Fennec engine — the only engine — with
AirLLM-only inference on the same machine and prompts, and writes
`results/engine_vs_airllm.{json,md}`. AirLLM is a training/inference library, not an engine
competitor; the comparison measures whether keeping the engine is worth it.

```bash
python scripts/bench_engine_vs_airllm.py --backend both --tokens 32 --airllm-tokens 4
```

AirLLM streams every layer from disk per token, so `--airllm-tokens` shrinks its budget independently;
a full 32-token sweep over three prompts takes hours. Two stock-AirLLM gaps are closed so the comparison
is a measurement rather than a crash: nested decoders (`model.language_model.layers`) get explicit layer
names, and a Qwen chat template is supplied when the container ships none.

Engine benchmarks: see benchmarks/ in the suite root.

## Context-payload benchmark

`scripts/bench_context_payload.py` replays one fixed agent transcript through an already-running
engine twice: the old payload (tool results sliced to 6,000 characters, duplicates repeated verbatim)
and `context.compact()`. Arms alternate so warm-up drift lands on both; both are greedy
with the same `max_tokens`. Writes `results/context_payload.{json,md}`.

Measured 2026-11-10 against the `qwen3.6-colibri` checkpoint (Vulkan, `--ctx 16384`,
`max_tokens 8`, `temperature 0`, 2 order-counterbalanced pairs):

| Arm | Payload chars | Prompt tokens | Median TTFT |
|---|---|---|---|
| old loop | 25,068 | 9,594 | 70.703 s |
| compacted | 19,066 | 6,995 | 51.141 s |
| **delta** | **−23.9 %** | **−2,599** | **−27.7 %** (−19.562 s) |

Source: `results/context_payload.json`. Compaction reduced a 52,543-character transcript to 19,413
(3 results elided, 1 deduplicated) under the default 48,000-character budget and 6,000-character result cap.

```bash
python scripts/bench_context_payload.py --base-url http://127.0.0.1:8197 --exact-usage
```

The script never starts or stops the engine, and needs only the standard library plus
`context.py`, so it runs on any Python 3.10+. Two pairs is a small sample: the
TTFT delta is a measurement on this host against Colibri, not a general latency claim and
not a Fennec figure.

## Tests

```bash
python -m unittest discover -s tests          # 105 checks across 4 files
```

Hermetic: sockets mocked, no engine, no model, no network. `test_pipeline_safety.py` imports `torch`
and `safetensors` (training deps required); the other two files do not.

## Prerequisites

Python 3.10+ with `torch`, `transformers`, `airllm`, `safetensors`, `huggingface_hub` for training;
`matplotlib` for graphs, `psutil` for the footprint benchmark. HTTP-only benches need just the standard
library plus `context.py`. Dependencies are declared by the suite hub's `npu` / `fine-tune`
features — no `requirements.txt`, `setup.py` or `pyproject.toml` here. Install with
`python setup.py install --features fine-tune` from
[fennec-environment-setup](https://github.com/Liso7150/fennec-environment-setup).

## Credits

[AirLLM](https://github.com/lyogavin/airllm) — layer-streamed inference and the streamed
LoRA trainer. [Fennec](https://github.com/Liso7150/fennec) — the native Rust engine that
quantises the merged shards and serves both models for the holdout.
[JustVugg/colibri](https://github.com/JustVugg/colibri) — the base idea Fennec was built from;
the upstream engine is retired and no longer resolved or launched. Part of the
[Fennec suite](https://github.com/Liso7150/fennec-environment-setup).
