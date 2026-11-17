"""Explicit merge/quant stages and a fresh, retention-only holdout evaluation.

--preflight-only resolves and checks paths without importing torch or loading
models. No promotion is performed; the evaluation compares the baseline against
the LoRA-merged candidate with the repo's own frozen holdout suite
(`bench_holdout.py`) against a Fennec engine it starts and stops itself.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

SCRIPTS = Path(__file__).resolve().parent
SUITE_ROOT = SCRIPTS.parent
HOLDOUT = SCRIPTS / "bench_holdout.py"
API_KEY = "local"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-root", default=os.environ.get("FENNEC_ENGINE_ROOT") or
                    os.environ.get("FENNEC_HOME") or str(Path.home() / "Fennec"))
    ap.add_argument("--models-dir", default=os.environ.get("FENNEC_MODELS") or
                    str(Path.home() / "fennec-models"))
    ap.add_argument("--results-dir", default=str(SUITE_ROOT / "results"))
    ap.add_argument("--engine-python", help="explicit runtime for engine tools; otherwise current interpreter")
    ap.add_argument("--base", help="bf16 source; defaults under models-dir")
    ap.add_argument("--adapter")
    ap.add_argument("--merged", help="existing merge input or NEW merge output")
    ap.add_argument("--baseline", help="baseline model dir; defaults to models-dir/qwen36-35b")
    ap.add_argument("--scaling", type=float, help="alpha/r, required for adapters without metadata")
    ap.add_argument("--stages", nargs="+", choices=("merge", "convert", "evaluate", "graph"),
                    default=["evaluate"], help="explicit opt-in; executed in pipeline order")
    ap.add_argument("--port", type=int, default=8185)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.engine_root:
        ap.error("set --engine-root or FENNEC_ENGINE_ROOT")
    if not 1 <= args.port <= 65535:
        ap.error("port must be in [1, 65535]")
    if "graph" in args.stages and "evaluate" not in args.stages:
        ap.error("graph stage requires evaluate; run it against an existing run directory instead")
    return args


def require_file(path):
    if not path.is_file():
        raise ValueError(f"required file missing: {path}")


def require_model(path):
    # What the Fennec engine actually reads: config.json, tokenizer.json and
    # shards (an index.json or loose *.safetensors files; Shards::open scans).
    for name in ("config.json", "tokenizer.json"):
        require_file(path / name)
    if (path / "model.safetensors.index.json").is_file():
        return
    if not any(p.suffix == ".safetensors" for p in path.iterdir()):
        raise ValueError(f"no safetensors shards in: {path}")


def require_new(path):
    if path.exists():
        raise ValueError(f"refusing existing output directory: {path}")


def require_free_port(port):
    # Binding (not HTTP health) also catches non-HTTP listeners and wildcard binds.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"port {port} is occupied/unavailable; refusing model launch") from exc


def engine_binary(root):
    return root / "target" / "release" / ("fennec.exe" if os.name == "nt" else "fennec")


def preflight(args):
    root = Path(args.engine_root).resolve()
    models = Path(args.models_dir).resolve()
    results = Path(args.results_dir).resolve()
    run_id = "eval-" + uuid.uuid4().hex
    binary = engine_binary(root)
    plan = {"engine_root": root, "binary": binary,
            "holdout_script": HOLDOUT,
            "models_dir": models, "results_dir": results,
            "run_dir": results / run_id, "run_id": run_id,
            "python": Path(sys.executable).resolve(),
            "engine_python": Path(args.engine_python or sys.executable).resolve(),
            "merge_script": SCRIPTS / "merge_lora_qwen36.py",
            "graph_script": SCRIPTS / "make_finetune_graph.py",
            "merged": Path(args.merged or models / ("Qwen3.6-35B-A3B-bf16-lora" +
                           ("-" + run_id if "merge" in args.stages else ""))).resolve(),
            "base": Path(args.base or models / "Qwen3.6-35B-A3B-bf16").resolve(),
            "adapter": Path(args.adapter or models / "qwen36-lora-adapter.pt").resolve(),
            "baseline": Path(args.baseline or models / "qwen36-35b").resolve(),
            "stages": [s for s in ("merge", "convert", "evaluate", "graph") if s in args.stages],
            "port": args.port, "scaling": args.scaling,
            "scope": "retention_only", "promotion_eligible": False,
            "tensor_validation": "deferred to merge; preflight never imports torch"}
    if "convert" in args.stages:
        # `fennec quant` writes the sidecar; the served candidate dir is the merge
        # output itself, so the sidecar stays out of the model tree.
        plan["sidecar"] = plan["run_dir"] / "finetune.fq4"
    if not root.is_dir() or not models.is_dir():
        raise ValueError("engine-root and models-dir must be existing directories")
    if results.exists() and not results.is_dir():
        raise ValueError("results-dir is not a directory")
    require_new(plan["run_dir"])
    require_file(plan["python"])
    require_file(plan["engine_python"])
    inputs = [plan[k] for k in ("base", "baseline", "adapter")]
    if "merge" in args.stages:
        require_new(plan["merged"])
        for other in inputs:
            if plan["merged"] == other or plan["merged"] in other.parents or other in plan["merged"].parents:
                raise ValueError(f"merge output conflicts with an input path: {plan['merged']}")
        require_file(plan["merge_script"])
        require_model(plan["base"])
        require_file(plan["adapter"])
        from merge_lora_qwen36 import adapter_metadata
        plan["scaling"], _ = adapter_metadata(plan["adapter"], plan["base"], args.scaling)
    if "convert" in args.stages:
        require_file(plan["binary"])
        if "merge" not in args.stages:
            require_model(plan["merged"])
    if "evaluate" in args.stages:
        require_file(plan["binary"])
        require_file(plan["holdout_script"])
        require_model(plan["baseline"])
        if "merge" not in args.stages:
            require_model(plan["merged"])
            if plan["merged"] == plan["baseline"]:
                raise ValueError("baseline and candidate must be different directories")
        require_free_port(args.port)
        if os.name == "nt":
            taskkill = shutil.which("taskkill")
            if not taskkill:
                raise ValueError("taskkill required to stop the owned server process tree")
            plan["taskkill"] = Path(taskkill).resolve()
    return plan


def run(command, plan):
    print("Launching: " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(plan["engine_root"]), check=True)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def stop_server(process, plan):
    if os.name == "nt":
        stopped = subprocess.run([str(plan["taskkill"]), "/PID", str(process.pid), "/T", "/F"],
                                 capture_output=True)
        if stopped.returncode and process.poll() is None:
            raise RuntimeError("could not stop owned server tree; refusing further launches")
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    process.wait(timeout=30)
    # Do not start the next model while the previous listener is still alive.
    require_free_port(plan["port"])


def serve_command(plan, model, label):
    command = [str(plan["binary"]), "serve", "--model", str(model), "--auto-tier",
               "--ctx", "8192", "--ngen", "700", "--host", "127.0.0.1",
               "--port", str(plan["port"]), "--api-key", API_KEY]
    if label == "finetune" and plan.get("sidecar"):
        command += ["--fq4", str(plan["sidecar"])]
    return command


def quant_command(plan):
    return [str(plan["binary"]), "quant", "--model", str(plan["merged"]), "--layers", "all",
            "--out", str(plan["sidecar"])]


def holdout_command(plan, model_id, label):
    return [str(plan["engine_python"]), "-B", str(plan["holdout_script"]),
            "--base", f"http://127.0.0.1:{plan['port']}/v1", "--model", model_id,
            "--api-key-env", "FENNEC_API_KEY",
            "--out", str(plan["run_dir"] / f"holdout_{label}.json"),
            "--max-tokens", "700"]


def evaluate_model(plan, model, label):
    port = plan["port"]
    require_free_port(port)
    model_id = model.name
    env = dict(os.environ, FENNEC_API_KEY=API_KEY, PYTHONDONTWRITEBYTECODE="1")
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (plan["run_dir"] / f"serve_{label}.log").open("xb") as log:
        require_free_port(port)
        process = subprocess.Popen(serve_command(plan, model, label), cwd=str(plan["engine_root"]),
                                   env=env, stdout=log, stderr=log, **options)
        try:
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited; see serve_{label}.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        ready = response.status == 200
                    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models",
                                                     headers={"Authorization": f"Bearer {API_KEY}"})
                    with urllib.request.urlopen(request, timeout=2) as response:
                        owned = model_id in {m["id"] for m in json.load(response)["data"]}
                    if ready and owned and process.poll() is None:
                        break
                except (OSError, ValueError, KeyError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("owned server did not become healthy in 300 seconds")
                time.sleep(1)
            run(holdout_command(plan, model_id, label), plan)
        finally:
            stop_server(process, plan)


def verdict(plan):
    sys.path.insert(0, str(SCRIPTS))
    from bench_holdout import compare_reports
    original = json.loads((plan["run_dir"] / "holdout_qwen36.json").read_text(encoding="utf-8"))
    candidate = json.loads((plan["run_dir"] / "holdout_finetune.json").read_text(encoding="utf-8"))
    return compare_reports(original, candidate)


def main(argv=None):
    args = parse_args(argv)
    plan = preflight(args)
    print(json.dumps(plan, indent=2, default=str), flush=True)
    if args.preflight_only:
        return
    plan["results_dir"].mkdir(parents=True, exist_ok=True)
    plan["run_dir"].mkdir()  # Atomic unique reservation; never reuse a run.
    write_json(plan["run_dir"] / "run_plan.json", plan)
    if "merge" in args.stages:
        require_new(plan["merged"])
        run([plan["python"], plan["merge_script"], "--base", plan["base"],
             "--adapter", plan["adapter"], "--out", plan["merged"], "--scaling", plan["scaling"]], plan)
    if "convert" in args.stages:
        require_model(plan["merged"])  # merge just produced it, or it is an input
        run(quant_command(plan), plan)
    if "evaluate" in args.stages:
        if plan.get("sidecar"):
            require_file(plan["sidecar"])
        evaluate_model(plan, plan["baseline"], "qwen36")
        evaluate_model(plan, plan["merged"], "finetune")
        write_json(plan["run_dir"] / "finetune_verdict.json", verdict(plan))
        print(json.dumps(json.loads((plan["run_dir"] / "finetune_verdict.json").read_text(encoding="utf-8")),
                         indent=2), flush=True)
    if "graph" in args.stages:
        run([plan["python"], plan["graph_script"], "--results-dir", plan["run_dir"]], plan)


if __name__ == "__main__":
    main()
