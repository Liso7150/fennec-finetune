"""Offline engine-resolution tests: Fennec is the only engine.

No engine is ever executed and no socket is connected: argv is inspected, never
run, and every fixture is a dummy file in a temporary directory. A retired
Colibri launcher (`c/coli`) on disk must never resolve, and no `COLI_*` knob is
ever written; stale `COLI_*` exports are still scrubbed defensively.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import bench_engine_vs_airllm as versus
import bench_tier_profiles as tiers
import engine_paths as paths

BINARY = paths.binary_names()[0]


class Spawn(Exception):
    """Stands in for a real launch: records argv, never starts a process."""


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        for target in ("socket.create_connection", "socket.getaddrinfo",
                       "socket.socket.connect", "socket.socket.connect_ex"):
            patcher = mock.patch(target, side_effect=AssertionError("network forbidden in engine tests"))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix="fennec-engine-test-")
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def touch(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
        return path

    def fennec_root(self, rel="target/release", name=BINARY):
        root = self.work / "fennec"
        root.mkdir(exist_ok=True)
        self.touch(root / rel / name)
        return root

    def launcher_only_root(self):
        """A retired Colibri checkout: c/coli exists, no Fennec binary anywhere."""
        root = self.work / "retired-checkout"
        self.touch(root / "c" / "coli")
        return root

    def both_root(self):
        root = self.launcher_only_root()
        self.touch(root / "target" / "release" / BINARY)
        return root


class ResolutionTests(OfflineTestCase):
    def test_fennec_resolves_even_beside_a_retired_launcher(self):
        root = self.both_root()
        engine = paths.resolve(root)
        self.assertEqual(engine.kind, paths.FENNEC)
        self.assertTrue(engine.is_fennec)
        self.assertEqual(engine.binary, root / "target" / "release" / BINARY)
        self.assertTrue((root / "c" / "coli").is_file(), "the stale launcher stays on disk, ignored")

    def test_every_probe_location_and_binary_name_is_honoured(self):
        for rel in ("target/release", "target/debug", "."):
            for name in paths.binary_names():
                with self.subTest(rel=rel, name=name):
                    root = self.work / f"probe-{rel.replace('/', '-')}-{name}"
                    root.mkdir()
                    self.touch(root / rel / name)
                    engine = paths.resolve(root)
                    self.assertEqual(engine.kind, paths.FENNEC)
                    self.assertEqual(engine.binary, (root / rel / name).resolve())

    def test_release_is_preferred_over_debug(self):
        root = self.fennec_root("target/debug")
        self.touch(root / "target" / "release" / BINARY)
        self.assertEqual(paths.resolve(root).binary, root / "target" / "release" / BINARY)

    def test_a_launcher_alone_raises_instead_of_resolving(self):
        root = self.launcher_only_root()
        with self.assertRaises(paths.EngineError) as caught:
            paths.resolve(root)
        message = str(caught.exception)
        self.assertIn("cargo build --release", message)
        self.assertIn("Fennec", message)
        self.assertNotIn("colibri", message.lower(), "no fallback engine is ever suggested")
        self.assertNotIn("c/coli", message)

    def test_no_engine_names_the_fennec_remedy(self):
        empty = self.work / "empty"
        empty.mkdir()
        with self.assertRaises(paths.EngineError) as caught:
            paths.resolve(empty)
        message = str(caught.exception)
        self.assertIn("cargo build --release", message)
        self.assertIn("--engine-root", message)
        self.assertIn("only engine", message)

    def test_default_root_prefers_env_then_fennec_and_never_colibri(self):
        home = self.work / "home"
        (home / "Fennec").mkdir(parents=True)
        (home / "Colibri").mkdir()
        legacy_home = self.work / "legacy-home"
        (legacy_home / "Colibri").mkdir(parents=True)
        with mock.patch.object(Path, "home", return_value=home):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(paths.default_engine_root(), str(home / "Fennec"))
            with mock.patch.dict(os.environ, {"FENNEC_HOME": "explicit-home"}, clear=True):
                self.assertEqual(paths.default_engine_root(), "explicit-home")
            with mock.patch.dict(os.environ, {"FENNEC_ENGINE_ROOT": "explicit",
                                              "FENNEC_HOME": "explicit-home"}, clear=True):
                self.assertEqual(paths.default_engine_root(), "explicit")
        with mock.patch.object(Path, "home", return_value=legacy_home):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(paths.default_engine_root(), str(legacy_home / "Fennec"),
                                 "a Colibri directory is never a fallback root")


class ServeArgvTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.model = self.work / "models" / "qwen36-35b"
        self.model.mkdir(parents=True)

    def test_fennec_serve_has_no_python_prefix(self):
        engine = paths.resolve(self.fennec_root())
        argv = paths.serve_command(engine, self.model, 8185)
        self.assertEqual(argv, [str(engine.binary), "serve", "--auto-tier",
                                "--model", str(self.model), "--host", "127.0.0.1", "--port", "8185"])
        self.assertNotIn(sys.executable, argv)
        self.assertNotIn("--api-key", argv, "Fennec reads the key from the environment")

    def test_fennec_serve_keeps_the_profile_knobs(self):
        engine = paths.resolve(self.fennec_root())
        argv = paths.serve_command(engine, self.model, 8185,
                                   extra=["--policy", "quality", "--ctx", "8192"])
        self.assertEqual(argv[:2], [str(engine.binary), "serve"])
        self.assertEqual(argv[argv.index("--port") + 1], "8185")
        self.assertEqual(argv[-4:], ["--policy", "quality", "--ctx", "8192"])
        self.assertNotIn("--api-key", argv)

    def test_fennec_command_never_takes_an_interpreter(self):
        engine = paths.resolve(self.fennec_root())
        self.assertEqual(paths.command(engine, ["doctor"]), [str(engine.binary), "doctor"])
        self.assertNotIn(sys.executable, paths.command(engine, ["info"]))


class EnvironmentTests(OfflineTestCase):
    def test_a_child_is_served_only_fennec_names(self):
        base = {"FENNEC_VULKAN": "1", "FENNEC_MODEL": "/models/m", "FENNEC_API_KEY": "local",
                "FENNEC_VK_TIER_HOST_GB": "4", "FENNEC_CUDA": "0", "FENNEC_METAL": "0",
                "PATH": "keep"}
        env = paths.engine_env(base=base)
        self.assertEqual(env, base)
        self.assertFalse(any(key.startswith("COLI_") for key in env),
                         "no knob is ever mirrored onto the retired spelling")

    def test_overrides_win_and_are_stringified_without_mirroring(self):
        env = paths.engine_env({"FENNEC_VULKAN": 1}, base={"FENNEC_VULKAN": "0"})
        self.assertEqual(env, {"FENNEC_VULKAN": "1"})

    def test_pin_writes_the_fennec_name_only(self):
        env = {}
        paths.pin(env, "VULKAN", "1")
        self.assertEqual(env, {"FENNEC_VULKAN": "1"})
        paths.pin(env, "VK_TIER_HOST_GB", 4)
        self.assertEqual(env, {"FENNEC_VULKAN": "1", "FENNEC_VK_TIER_HOST_GB": "4"})

    def test_env_alias_ignores_the_retired_spelling(self):
        self.assertIsNone(paths.env_alias("MODEL", {"COLI_MODEL": "old"}))
        self.assertEqual(paths.env_alias("MODEL", {"FENNEC_MODEL": "new", "COLI_MODEL": "old"}), "new")
        self.assertIsNone(paths.env_alias("MODEL", {}))

    def test_stripping_removes_both_spellings_but_keeps_the_rest(self):
        env = paths.strip_engine_vars({"COLI_VULKAN": "1", "FENNEC_VULKAN": "1",
                                       "COLI_VK_TIER_GB": "4", "FENNEC_VK_TIER_HOST_GB": "4",
                                       "FENNEC_MODELS": "/models", "PYTHONUTF8": "1"})
        self.assertEqual(env, {"FENNEC_MODELS": "/models", "PYTHONUTF8": "1"})


class BenchCallSiteTests(OfflineTestCase):
    """The HTTP benches launch Fennec, without a real spawn."""

    def setUp(self):
        super().setUp()
        self.model = self.work / "models" / "qwen36-35b"
        self.model.mkdir(parents=True)
        self.out = self.work / "results"
        self.out.mkdir()

    def launch(self, call, module):
        with mock.patch.object(module, "RESULTS", self.out), \
                mock.patch.object(module, "free_port", return_value=8199), \
                mock.patch.object(module.subprocess, "Popen", side_effect=Spawn) as popen:
            with self.assertRaises(Spawn):
                call()
        return popen.call_args

    def test_tier_profiles_launch_fennec_directly(self):
        engine = paths.resolve(self.fennec_root())
        args = argparse.Namespace(model=str(self.model), ctx=8192, ngen=128, engine_root=str(engine.root))
        extra_env, extra_args = tiers.PROFILES[0][1], tiers.PROFILES[0][2]
        call = self.launch(lambda: tiers.run_profile("pool 4 GB, gpu auto", extra_env, extra_args, args, engine),
                           tiers)
        argv = call.args[0]
        self.assertEqual(argv[:2], [str(engine.binary), "serve"])
        self.assertNotIn(sys.executable, argv)
        self.assertNotIn("--api-key", argv)
        self.assertEqual(argv[argv.index("--model") + 1], str(self.model))
        self.assertEqual(argv[argv.index("--port") + 1], "8199")
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertIn("--auto-tier", argv)
        self.assertEqual(argv.count("--auto-tier"), 1, "the helper owns --auto-tier")
        env = call.kwargs["env"]
        self.assertEqual(env["FENNEC_VK_TIER_HOST_GB"], "4")
        self.assertEqual(env["FENNEC_API_KEY"], "local")
        self.assertNotIn("COLI_VK_TIER_HOST_GB", env, "the native child reads one spelling")
        self.assertNotIn("COLI_API_KEY", env)
        self.assertEqual(call.kwargs["cwd"], str(engine.root))

    def test_tier_profiles_refuse_a_launcher_only_root_without_spawning(self):
        root = self.launcher_only_root()
        with mock.patch.object(tiers.subprocess, "Popen", side_effect=AssertionError("must not launch")) as popen:
            with self.assertRaises(SystemExit) as caught:
                tiers.main(["--engine-root", str(root), "--model", str(self.model)])
        popen.assert_not_called()
        self.assertIn("cargo build --release", str(caught.exception))

    def test_tier_profiles_strip_every_spelling_and_pin_one_for_a_native_child(self):
        engine = paths.resolve(self.fennec_root())
        args = argparse.Namespace(model=str(self.model), ctx=8192, ngen=128, engine_root=str(engine.root))
        extra_env, extra_args = tiers.PROFILES[0][1], tiers.PROFILES[0][2]
        with mock.patch.dict(os.environ, {"COLI_VK_TIER_HOST_GB": "13", "FENNEC_VULKAN": "0",
                                          "FENNEC_API_KEY": "ambient"}):
            call = self.launch(lambda: tiers.run_profile("pool 4 GB, gpu auto", extra_env, extra_args, args, engine),
                               tiers)
        env = call.kwargs["env"]
        # The ambient shell may carry either spelling; both are stripped, and the
        # native child is then pinned under Fennec's name only.
        self.assertEqual(env["FENNEC_VK_TIER_HOST_GB"], "4", "the profile must win over the ambient value")
        self.assertEqual(env["FENNEC_VULKAN"], "1", "the profile's pin replaces the ambient one")
        self.assertEqual(env["FENNEC_API_KEY"], "local")
        self.assertNotIn("COLI_VK_TIER_HOST_GB", env)
        self.assertNotIn("COLI_VULKAN", env)
        self.assertNotIn("COLI_API_KEY", env)

    def test_engine_vs_airllm_launches_fennec_directly(self):
        engine = paths.resolve(self.fennec_root())
        args = argparse.Namespace(engine_root=str(engine.root), engine_api_key="local",
                                  model=str(self.model), tokens=32, backend_engine="vulkan", port=8197,
                                  startup_timeout=1)
        call = self.launch(lambda: versus.bench_engine(args), versus)
        argv = call.args[0]
        self.assertEqual(argv[:2], [str(engine.binary), "serve"])
        self.assertNotIn(sys.executable, argv)
        self.assertNotIn("--api-key", argv)
        env = call.kwargs["env"]
        self.assertEqual(env["FENNEC_API_KEY"], "local")
        self.assertEqual(env["FENNEC_VULKAN"], "1")
        self.assertNotIn("COLI_API_KEY", env, "the native child reads one spelling")
        self.assertNotIn("COLI_VULKAN", env)

    def test_engine_vs_airllm_refuses_a_launcher_only_root_without_spawning(self):
        root = self.launcher_only_root()
        args = argparse.Namespace(engine_root=str(root), engine_api_key="local",
                                  model=str(self.model), tokens=32, backend_engine="vulkan", port=8197,
                                  startup_timeout=1)
        with mock.patch.object(versus.subprocess, "Popen", side_effect=AssertionError("must not launch")) as popen:
            report = versus.bench_engine(args)
        popen.assert_not_called()
        self.assertEqual(report["status"], "error")
        self.assertIn("cargo build --release", report["error"])
        self.assertNotIn("colibri", report["error"].lower())

    def test_engine_vs_airllm_reports_a_missing_engine_without_spawning(self):
        empty = self.work / "empty"
        empty.mkdir()
        args = argparse.Namespace(engine_root=str(empty), engine_api_key="local",
                                  model=str(self.model), tokens=32, backend_engine="vulkan", port=8197,
                                  startup_timeout=1)
        with mock.patch.object(versus.subprocess, "Popen", side_effect=AssertionError("must not launch")) as popen:
            report = versus.bench_engine(args)
        popen.assert_not_called()
        self.assertEqual(report["status"], "error")
        self.assertIn("cargo build --release", report["error"])

    def test_engine_vs_airllm_refuses_the_retired_interpreter_flag(self):
        with mock.patch.object(versus.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            with self.assertRaises(SystemExit) as caught:
                versus.main(["--engine-python", "interp"])
        self.assertNotEqual(caught.exception.code, 0)

    def test_the_report_names_fennec_the_only_engine(self):
        self.assertEqual(versus.engine_label({"engine": "fennec"}), "Fennec engine")
        self.assertEqual(versus.engine_label({}), "Fennec engine", "a failed resolve is still a Fennec report")
        self.assertEqual(versus.engine_label({"engine": "colibri"}), "Fennec engine",
                         "no Colibri label is ever emitted")


if __name__ == "__main__":
    unittest.main()