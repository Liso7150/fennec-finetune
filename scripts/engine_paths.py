"""Shared engine resolution: Fennec (native Rust) is the only engine.

Fennec self-launches (`target/release/fennec[.exe] serve ...`) and speaks the
OpenAI-compatible HTTP contract on the default port. The upstream Colibri
launcher is retired: it is never resolved, launched, or mirrored to.

This mirrors fennec-core's convention (engine.py::_engine_binary):
probe `target/release`, `target/debug`, `.` for the fennec binary. Stdlib only -
preflight must not import torch.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

FENNEC = "fennec"

# The engine knobs these harnesses place. The native binary reads `FENNEC_<NAME>`
# and nothing else. The retired `COLI_<NAME>` spelling is only ever scrubbed, so
# a stale shell export cannot leak into a measurement run.
KNOBS = ("MODEL", "API_KEY", "VULKAN", "METAL", "CUDA",
         "VK_TIER_HOST_GB", "VK_TIER_GB", "VK_DEV2")
FENNEC_VARS = frozenset("FENNEC_" + name for name in KNOBS)

DEFAULT_PORT = 8184


class EngineError(RuntimeError):
    """No usable engine under the requested root."""


class Engine(NamedTuple):
    kind: str
    binary: Path
    root: Path

    @property
    def is_fennec(self) -> bool:
        return self.kind == FENNEC


def binary_names() -> tuple:
    return ("fennec.exe", "fennec") if os.name == "nt" else ("fennec",)


def default_engine_root() -> str:
    """FENNEC_ENGINE_ROOT / FENNEC_HOME, else ~/Fennec."""
    for variable in ("FENNEC_ENGINE_ROOT", "FENNEC_HOME"):
        value = os.environ.get(variable)
        if value:
            return value
    return str(Path.home() / "Fennec")


def resolve(root) -> Engine:
    """Find the Fennec binary; there is no fallback engine."""
    root = Path(root).expanduser().resolve()
    for rel in ("target/release", "target/debug", "."):
        for name in binary_names():
            candidate = root / rel / name
            if candidate.is_file():
                return Engine(FENNEC, candidate, root)
    raise EngineError(
        f"no engine found under {root}: Fennec is the only engine; build it with "
        f"cargo build --release so {root / 'target' / 'release' / binary_names()[0]} "
        "exists, or point --engine-root at a Fennec checkout")


def command(engine: Engine, args) -> list:
    """The native binary self-launches; no interpreter prefix exists."""
    return [str(engine.binary), *[str(arg) for arg in args]]


def serve_command(engine: Engine, model, port, *, host="127.0.0.1", extra=()) -> list:
    """`serve --auto-tier --model <dir> --host <host> --port <n>` for Fennec.

    Fennec takes the key from `FENNEC_API_KEY`, which the caller pins; there is
    no --api-key flag to pass.
    """
    args = ["serve", "--auto-tier", "--model", str(model), "--host", host, "--port", str(port)]
    args += [str(arg) for arg in extra]
    return command(engine, args)


def env_alias(name, env=None):
    """The value an operator exported for one knob: `FENNEC_<name>` only."""
    env = os.environ if env is None else env
    return env.get("FENNEC_" + name)


def engine_env(overrides=None, *, base=None) -> dict:
    """The child environment: overrides applied over the base. Fennec reads
    `FENNEC_*` alone; nothing is mirrored onto the retired `COLI_*` names."""
    env = dict(os.environ if base is None else base)
    if overrides:
        env.update({key: str(value) for key, value in overrides.items()})
    return env


def strip_engine_vars(env) -> dict:
    """Drop every engine knob under either spelling, so a profile starts clean.

    The retired `COLI_*` name is scrubbed defensively: a stale `COLI_VULKAN` in
    the shell must not be able to change what a measurement run places.
    """
    return {key: value for key, value in env.items()
            if not key.startswith("COLI_") and key not in FENNEC_VARS}


def pin(env, name, value) -> None:
    """Set one knob under the single name the native child reads: `FENNEC_<NAME>`."""
    env["FENNEC_" + name] = str(value)