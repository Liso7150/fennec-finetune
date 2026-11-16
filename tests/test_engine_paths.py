"""Offline engine-resolution tests: Fennec first, Colibri fallback, loud refusals.

No engine is ever executed and no socket is connected: argv is inspected, never
run, and every fixture is a dummy file in a temporary directory.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
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
import run_finetune_eval as evaluation

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

    def colibri_root(self):
        root = self.work / "colibri"
        self.touch(root / "c" / "coli")
        return root

    def both_root(self):
        root = self.colibri_root()
        self.touch(root / "target" / "release" / BINARY)
        return root


class ResolutionTests(OfflineTestCase):
    def test_fennec_wins_over_colibri_in_the_same_root(self):
        root = self.both_root()
        engine = paths.resolve(root)
        self.assertEqual(engine.kind, paths.FENNEC)
        self.assertEqual(engine.binary, root / "target" / "release" / BINARY)
        self.assertTrue((root / "c" / "coli").is_file(), "the fallback must stay on disk")

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

    def test_colibri_launcher_is_the_fallback(self):
        root = self.colibri_root()
        engine = paths.resolve(root)
        self.assertEqual(engine.kind, paths.COLIBRI)
        self.assertEqual(engine.binary, root / "c" / "coli")
        self.assertEqual(engine.root, root.resolve())

    def test_no_engine_names_both_remedies(self):
        empty = self.work / "empty"
        empty.mkdir()
        with self.assertRaises(paths.EngineError) as caught:
            paths.resolve(empty)
        message = str(caught.exception)
        self.assertIn("cargo build --release", message)
        self.assertIn("c/coli", message)
        self.assertIn("--engine-root", message)

    def test_default_root_prefers_env_then_fennec_then_colibri(self):
        home = self.work / "home"
        (home / "Fennec").mkdir(parents=True)
        (home / "Colibri").mkdir()
        legacy_home = self.work / "legacy-home"
        (legacy_home / "Colibri").mkdir(parents=True)
        with mock.patch.object(Path, "home", return_value=home):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(paths.default_engine_root(), str(home / "Fennec"))
            with mock.patch.dict(os.environ, {"FENNEC_HOME": str(home / "Colibri")}, clear=True):
                self.assertEqual(paths.default_engine_root(), str(home / "Colibri"))
            with mock.patch.dict(os.environ, {"FENNEC_ENGINE_ROOT": "explicit",
                                              "FENNEC_HOME": str(home / "Colibri")}, clear=True):
                self.assertEqual(paths.default_engine_root(), "explicit")
        with mock.patch.object(Path, "home", return_value=legacy_home):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(paths.default_engine_root(), str(legacy_home / "Colibri"))


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
                                   extra=["--policy", "quality", "--ctx", "8192"], api_key="local")
        self.assertEqual(argv[:2], [str(engine.binary), "serve"])
        self.assertEqual(argv[argv.index("--port") + 1], "8185")
        self.assertEqual(argv[-4:], ["--policy", "quality", "--ctx", "8192"])
        self.assertNotIn("--api-key", argv)

    def test_colibri_serve_keeps_the_interpreter_prefix_and_api_key(self):
        root = self.colibri_root()
        engine = paths.resolve(root)
        argv = paths.serve_command(engine, self.model, 8185, api_key="local", python="interp")
        self.assertEqual(argv, ["interp", str(root / "c" / "coli"), "serve", "--auto-tier",
                                "--model", str(self.model), "--host", "127.0.0.1",
                                "--port", "8185", "--api-key", "local"])

    def test_colibri_prefix_defaults_to_the_current_interpreter(self):
        engine = paths.resolve(self.colibri_root())
        argv = paths.command(engine, ["info"])
        self.assertEqual(argv, [sys.executable, str(engine.binary), "info"])

    def test_fennec_command_never_takes_an_interpreter(self):
        engine = paths.resolve(self.fennec_root())
        self.assertEqual(paths.command(engine, ["doctor"], python="interp"),
                         [str(engine.binary), "doctor"])


class EnvironmentTests(OfflineTestCase):
    def test_a_colibri_child_is_served_the_names_its_launcher_reads(self):
        base = {"FENNEC_VULKAN": "1", "FENNEC_MODEL": "/models/m", "FENNEC_API_KEY": "local",
                "FENNEC_VK_TIER_HOST_GB": "4", "FENNEC_CUDA": "0", "FENNEC_METAL": "0",
                "PATH": "keep"}
        env = paths.engine_env(base=base, legacy=True)
        for name in ("VULKAN", "MODEL", "API_KEY", "VK_TIER_HOST_GB", "CUDA", "METAL"):
            self.assertEqual(env["COLI_" + name], base["FENNEC_" + name], name)
        self.assertEqual(env["PATH"], "keep")

    def test_a_native_child_is_served_only_fennec_names(self):
        env = paths.engine_env(base={"FENNEC_VULKAN": "1", "FENNEC_MODEL": "/models/m"})
        self.assertEqual(env, {"FENNEC_VULKAN": "1", "FENNEC_MODEL": "/models/m"})

    def test_an_explicit_coli_value_is_not_overwritten(self):
        env = paths.engine_env(base={"COLI_VK_TIER_HOST_GB": "13", "FENNEC_VK_TIER_HOST_GB": "4"},
                               legacy=True)
        self.assertEqual(env["COLI_VK_TIER_HOST_GB"], "13")

    def test_overrides_win_and_are_stringified(self):
        env = paths.engine_env({"COLI_VULKAN": 1}, base={"FENNEC_VULKAN": "0"})
        self.assertEqual(env["COLI_VULKAN"], "1")

    def test_a_pinned_override_is_mirrored_for_a_colibri_child(self):
        env = paths.engine_env({"FENNEC_VULKAN": 1}, base={}, legacy=True)
        self.assertEqual(env["COLI_VULKAN"], "1")

    def test_pin_writes_the_legacy_name_only_for_a_colibri_child(self):
        env = {}
        paths.pin(env, "VULKAN", "1")
        self.assertEqual(env, {"FENNEC_VULKAN": "1"})
        env = {}
        paths.pin(env, "VULKAN", "1", legacy=True)
        self.assertEqual(env, {"FENNEC_VULKAN": "1", "COLI_VULKAN": "1"})

    def test_env_alias_reads_either_spelling(self):
        self.assertEqual(paths.env_alias("MODEL", {"COLI_MODEL": "old"}), "old")
        self.assertEqual(paths.env_alias("MODEL", {"FENNEC_MODEL": "new", "COLI_MODEL": "old"}), "new")
        self.assertIsNone(paths.env_alias("MODEL", {}))

    def test_stripping_removes_both_spellings_but_keeps_the_rest(self):
        env = paths.strip_engine_vars({"COLI_VULKAN": "1", "FENNEC_VULKAN": "1",
                                       "COLI_VK_TIER_GB": "4", "FENNEC_VK_TIER_HOST_GB": "4",
                                       "FENNEC_MODELS": "/models", "PYTHONUTF8": "1"})
        self.assertEqual(env, {"FENNEC_MODELS": "/models", "PYTHONUTF8": "1"})


class ColibriOnlyEvalTests(OfflineTestCase):
    """The legacy convert/evaluate path must refuse Fennec, never fake or skip it."""

    def setUp(self):
        super().setUp()
        self.models = self.work / "models"
        self.models.mkdir()
        self.results = self.work / "results"
        self.argv = ["--engine-root", str(self.fennec_root()), "--models-dir", str(self.models),
                     "--results-dir", str(self.results), "--port", "8185"]

    def refuse(self, stages, *expected):
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("must not launch")) as launch:
            with self.assertRaises(paths.ColibriOnlyError) as caught:
                evaluation.preflight(evaluation.parse_args(self.argv + ["--stages", *stages]))
            launch.assert_not_called()
        message = str(caught.exception)
        for fragment in expected:
            self.assertIn(fragment, message)
        self.assertIn("--engine-root", message)
        self.assertIn("Colibri checkout", message)
        self.assertFalse(self.results.exists(), "a refusal must not write anything")
        return message

    def test_evaluate_refuses_a_fennec_root_and_names_every_missing_capability(self):
        self.refuse(["evaluate"], "bench_best_model.py", "qwen36.exe", "qwen36_meta.json", "COLI_ENGINE")

    def test_convert_refuses_a_fennec_root(self):
        self.refuse(["convert"], "convert_qwen36.py")

    def test_preflight_only_refuses_too(self):
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("must not launch")) as launch:
            with self.assertRaises(paths.ColibriOnlyError):
                evaluation.main(self.argv + ["--preflight-only"])
            launch.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_the_refusal_is_an_engine_error_subclass(self):
        self.assertTrue(issubclass(paths.ColibriOnlyError, paths.EngineError))

    def test_require_colibri_is_a_noop_for_a_colibri_root(self):
        engine = paths.resolve(self.colibri_root())
        self.assertIsNone(paths.require_colibri(engine, "anything at all"))


class ColibriPreflightTests(OfflineTestCase):
    """The Colibri path stays fully working, including the legacy evaluate stage."""

    def setUp(self):
        super().setUp()
        self.engine = self.colibri_root()
        self.touch(self.engine / "c" / "tools" / "convert_qwen36.py")
        self.touch(self.engine / "c" / ("qwen36.exe" if os.name == "nt" else "qwen36"))
        self.bench = self.engine / "scripts" / "bench_best_model.py"
        self.bench.parent.mkdir(parents=True, exist_ok=True)
        self.bench.write_text('RESULTS = "must-not-use-engine-results"\n', encoding="utf-8")
        self.models = self.work / "models"
        for name, converted in (("Qwen3.6-35B-A3B-bf16", False), ("qwen36-35b", True),
                                ("qwen36-35b-lora", True)):
            directory = self.models / name
            directory.mkdir(parents=True)
            for file in ("config.json", "tokenizer.json",
                         "qwen36_meta.json" if converted else "model.safetensors.index.json"):
                (directory / file).write_text("{}", encoding="utf-8")
        (self.models / "qwen36-lora-adapter.pt").write_bytes(b"synthetic-not-loaded")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.argv = ["--engine-root", str(self.engine), "--models-dir", str(self.models),
                     "--results-dir", str(self.work / "results"), "--port", str(self.port)]

    def test_evaluate_preflight_still_resolves_the_legacy_paths(self):
        plan = evaluation.preflight(evaluation.parse_args(self.argv))
        self.assertEqual(plan["engine_kind"], paths.COLIBRI)
        self.assertEqual(plan["launcher"], self.engine / "c" / "coli")
        self.assertEqual(plan["binary"], self.engine / "c" / ("qwen36.exe" if os.name == "nt" else "qwen36"))
        self.assertEqual(plan["bench_script"], self.bench)
        self.assertEqual(plan["convert_script"], self.engine / "c" / "tools" / "convert_qwen36.py")
        self.assertFalse((self.work / "results").exists())

    def test_the_owned_server_argv_keeps_the_interpreter_and_model_id(self):
        plan = evaluation.preflight(evaluation.parse_args(self.argv))
        engine = paths.Engine(plan["engine_kind"], Path(plan["launcher"]), Path(plan["engine_root"]))
        argv = paths.command(engine, ["serve", "--model", "m", "--model-id", "run-qwen36"],
                             python=plan["engine_python"])
        self.assertEqual(argv[:3], [str(plan["engine_python"]), str(self.engine / "c" / "coli"), "serve"])
        self.assertEqual(argv[-1], "run-qwen36")


class BenchCallSiteTests(OfflineTestCase):
    """The HTTP benches launch whichever engine was resolved, without a real spawn."""

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
        self.assertEqual(argv[argv.index("--model") + 1], str(self.model))
        self.assertEqual(argv[argv.index("--port") + 1], "8199")
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertIn("--auto-tier", argv)
        self.assertEqual(argv.count("--auto-tier"), 1, "the helper owns --auto-tier")
        env = call.kwargs["env"]
        self.assertEqual(env["FENNEC_VK_TIER_HOST_GB"], "4")
        self.assertEqual(env["FENNEC_API_KEY"], "local")
        self.assertNotIn("COLI_VK_TIER_HOST_GB", env, "a native child reads one spelling")
        self.assertNotIn("COLI_API_KEY", env)
        self.assertEqual(call.kwargs["cwd"], str(engine.root))

    def test_tier_profiles_keep_the_colibri_launcher(self):
        engine = paths.resolve(self.colibri_root())
        args = argparse.Namespace(model=str(self.model), ctx=8192, ngen=128, engine_root=str(engine.root))
        extra_env, extra_args = tiers.PROFILES[0][1], tiers.PROFILES[0][2]
        call = self.launch(lambda: tiers.run_profile("pool 4 GB, gpu auto", extra_env, extra_args, args, engine),
                           tiers)
        argv = call.args[0]
        self.assertEqual(argv[:3], [sys.executable, str(engine.binary), "serve"])
        self.assertEqual(argv[argv.index("--api-key") + 1], "local")
        self.assertEqual(argv[argv.index("--ctx") + 1], "8192")
        env = call.kwargs["env"]
        self.assertEqual(env["COLI_VK_TIER_HOST_GB"], "4", "the launcher reads its own name")
        self.assertEqual(env["FENNEC_VK_TIER_HOST_GB"], "4")

    def test_tier_profiles_strip_every_spelling_and_pin_one_for_a_native_child(self):
        engine = paths.resolve(self.fennec_root())
        args = argparse.Namespace(model=str(self.model), ctx=8192, ngen=128, engine_root=str(engine.root))
        extra_env, extra_args = tiers.PROFILES[0][1], tiers.PROFILES[0][2]
        with mock.patch.dict(os.environ, {"COLI_VK_TIER_HOST_GB": "13", "FENNEC_VULKAN": "0",
                                          "FENNEC_API_KEY": "ambient"}):
            call = self.launch(lambda: tiers.run_profile("pool 4 GB, gpu auto", extra_env, extra_args, args, engine),
                               tiers)
        env = call.kwargs["env"]
        # The ambient shell may carry either spelling; both are stripped, and a
        # native child is then pinned under Fennec's name only.
        self.assertEqual(env["FENNEC_VK_TIER_HOST_GB"], "4", "the profile must win over the ambient value")
        self.assertEqual(env["FENNEC_VULKAN"], "1", "the profile's pin replaces the ambient one")
        self.assertEqual(env["FENNEC_API_KEY"], "local")
        self.assertNotIn("COLI_VK_TIER_HOST_GB", env)
        self.assertNotIn("COLI_VULKAN", env)
        self.assertNotIn("COLI_API_KEY", env)

    def test_engine_vs_airllm_launches_fennec_directly(self):
        engine = paths.resolve(self.fennec_root())
        args = argparse.Namespace(engine_root=str(engine.root), engine_python=None, engine_api_key="local",
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
        self.assertNotIn("COLI_API_KEY", env, "a native child reads one spelling")
        self.assertNotIn("COLI_VULKAN", env)

    def test_engine_vs_airllm_keeps_the_colibri_launcher(self):
        engine = paths.resolve(self.colibri_root())
        args = argparse.Namespace(engine_root=str(engine.root), engine_python="interp", engine_api_key="local",
                                  model=str(self.model), tokens=32, backend_engine="vulkan", port=8197,
                                  startup_timeout=1)
        call = self.launch(lambda: versus.bench_engine(args), versus)
        argv = call.args[0]
        self.assertEqual(argv[:3], ["interp", str(engine.binary), "serve"])
        self.assertEqual(argv[argv.index("--api-key") + 1], "local")
        env = call.kwargs["env"]
        self.assertEqual(env["COLI_VULKAN"], "1", "the launcher reads the name it was written against")
        self.assertEqual(env["COLI_API_KEY"], "local")

    def test_engine_vs_airllm_reports_a_missing_engine_without_spawning(self):
        empty = self.work / "empty"
        empty.mkdir()
        args = argparse.Namespace(engine_root=str(empty), engine_python=None, engine_api_key="local",
                                  model=str(self.model), tokens=32, backend_engine="vulkan", port=8197,
                                  startup_timeout=1)
        with mock.patch.object(versus.subprocess, "Popen", side_effect=AssertionError("must not launch")) as popen:
            report = versus.bench_engine(args)
        popen.assert_not_called()
        self.assertEqual(report["status"], "error")
        self.assertIn("cargo build --release", report["error"])

    def test_the_report_names_the_engine_that_ran(self):
        self.assertEqual(versus.engine_label({"engine": "fennec"}), "Fennec engine")
        self.assertEqual(versus.engine_label({"engine": "colibri"}), "Colibri engine")
        self.assertEqual(versus.engine_label({}), "Colibri engine", "archived reports keep their label")


if __name__ == "__main__":
    unittest.main()
