"""Tests for write_clean_host_run.py.

When the ModelJars-mirroring validator (assemble_component_report.validate_clean_host_run, merged to
integrallis/models main in PR #189) is reachable through `git show origin/main:...`, a passing record
is also run through it with the revision placeholder substituted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import write_clean_host_run as writer  # noqa: E402

JAVA_25 = (
    'openjdk version "25.0.4.1" 2026-07-21 LTS\n'
    "OpenJDK Runtime Environment Temurin-25.0.4.1+1 (build 25.0.4.1+1-LTS)\n"
    "OpenJDK 64-Bit Server VM Temurin-25.0.4.1+1 (build 25.0.4.1+1-LTS, mixed mode, sharing)\n"
)
FRESH = "absent /root/.m2\nabsent /root/.jbang\nabsent /root/.gradle\n"
LOG_PATH = "benchmark-results/2026-09-16-granite-answerability-alora/release-pilot2/clean-host/evidence/clean-host-output.log"
LIBRARY_SHA256 = "b74d6e386d8494455b10698c1042e8d9e746cc60c8dce3b936b7cb5c203589d8"


def log_text(backend: str, *, native: bool | None = None, cases: int = 6) -> str:
    """Builds a program log of the shape AnswerabilityCleanHost emits, for one backend."""
    if native is None:
        native = backend == "rust-ffm"
    lines = []
    if native:
        lines.append(
            "native-library backend=rust-ffm platform=linux-x86_64 abi=5"
            f" file=libjmodels_kernels.so sha256={LIBRARY_SHA256}"
            " source=bundled-classpath kernelPlan=rust-ffm-v13"
        )
    injected = "true" if backend == "rust-ffm" else "false"
    kernel = "rust-ffm-quantized-v13" if backend == "rust-ffm" else "vector-api"
    lines.append(
        f"loaded backend={backend} adapterSha256={'a' * 64} invocation=\"marker\""
        f" injectedGroupedProjections={injected} matrixKernel={kernel} millis=42"
    )
    for index in range(cases):
        lines.append(
            f"CASE PASS backend={backend} suite=squad-v2-dev id=case{index}"
            " byteIdentical=true millis=1"
        )
    lines.append(f"PASS cases={cases} passed={cases}")
    return "\n".join(lines) + "\n"


class WriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.classpath = self.dir / "resolved-classpath.txt"
        self.classpath.write_text("aa  a.jar\nbb  b.jar\n", encoding="utf-8")
        self.log = self.dir / "clean-host-output.log"
        self.log.write_text(log_text("pure-java"), encoding="utf-8")

    def build(self, **overrides):
        arguments = dict(
            started_at="2026-09-17T10:00:00Z",
            completed_at="2026-09-17T10:07:00Z",
            exit_code=0,
            models_version="0.3.42",
            backend="pure-java",
            command=["jbang", "AnswerabilityCleanHost.java", "--store", "/opt/store"],
            java_version_output=JAVA_25,
            fresh_check=FRESH,
            classpath_file=self.classpath,
            output_log=self.log,
            output_log_repo_path=LOG_PATH,
        )
        arguments.update(overrides)
        return writer.build(**arguments)

    def rust(self, **overrides):
        self.log.write_text(overrides.pop("log", log_text("rust-ffm")), encoding="utf-8")
        return self.build(
            backend="rust-ffm",
            command=[
                "jbang",
                "--deps",
                "com.integrallis:backend-native:0.3.42",
                "AnswerabilityCleanHost.java",
                "--backend",
                "rust-ffm",
                "--store",
                "/opt/store",
            ],
            **overrides,
        )

    def test_passing_run_has_the_gate_shape(self) -> None:
        run = self.build()
        self.assertEqual(
            set(run),
            {
                "pass", "freshMachine", "externalInference", "javaMajor", "exitCode", "modelsVersion",
                "backend", "nativeLibrary", "startedAt", "completedAt", "command",
                "resolvedClasspathSha256", "outputLog",
            },
        )
        self.assertIs(run["pass"], True)
        self.assertIs(run["freshMachine"], True)
        self.assertIs(run["externalInference"], False)
        self.assertEqual(run["javaMajor"], 25)
        self.assertEqual(run["exitCode"], 0)
        self.assertEqual(run["backend"], "pure-java")
        self.assertIsNone(run["nativeLibrary"])
        self.assertEqual(run["resolvedClasspathSha256"], hashlib.sha256(b"aa  a.jar\nbb  b.jar\n").hexdigest())
        self.assertEqual(run["outputLog"]["sha256"], hashlib.sha256(self.log.read_bytes()).hexdigest())
        self.assertEqual(run["outputLog"]["sizeBytes"], len(self.log.read_bytes()))
        self.assertEqual(
            run["outputLog"]["uri"],
            f"https://raw.githubusercontent.com/integrallis/models/<EVIDENCE_REVISION>/{LOG_PATH}",
        )

    def test_rust_ffm_run_records_the_native_library_identity(self) -> None:
        run = self.rust()
        self.assertIs(run["pass"], True)
        self.assertEqual(run["backend"], "rust-ffm")
        self.assertEqual(
            run["nativeLibrary"],
            {
                "platform": "linux-x86_64",
                "abi": 5,
                "library": "libjmodels_kernels.so",
                "sha256": LIBRARY_SHA256,
                "source": "bundled-classpath",
                "kernelPlan": "rust-ffm-v13",
            },
        )

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.build(backend="llama-cpp")
        with self.assertRaises(ValueError):
            self.build(backend="")

    def test_declared_backend_must_match_the_log(self) -> None:
        # A pure-Java log relabelled as a Rust run must not pass.
        run = self.build(backend="rust-ffm")
        self.assertIs(run["pass"], False)
        self.assertIsNone(run["nativeLibrary"])
        self.log.write_text(log_text("rust-ffm"), encoding="utf-8")
        self.assertIs(self.build(backend="pure-java")["pass"], False)

    def test_log_without_a_loaded_backend_line_fails(self) -> None:
        self.log.write_text("CASE PASS ...\nPASS cases=6 passed=6\n", encoding="utf-8")
        run = self.build()
        self.assertIsNone(writer.logged_backend(self.log.read_text(encoding="utf-8")))
        self.assertIs(run["pass"], False)

    def test_every_case_line_must_carry_the_selected_backend(self) -> None:
        mixed = log_text("rust-ffm").replace(
            "CASE PASS backend=rust-ffm suite=squad-v2-dev id=case3",
            "CASE PASS backend=pure-java suite=squad-v2-dev id=case3",
        )
        self.assertIs(self.rust(log=mixed)["pass"], False)

    def test_case_line_count_must_match_the_expected_cases(self) -> None:
        self.log.write_text(log_text("pure-java", cases=5), encoding="utf-8")
        self.assertIs(self.build()["pass"], False)

    def test_rust_ffm_without_a_native_library_line_fails(self) -> None:
        self.assertIs(self.rust(log=log_text("rust-ffm", native=False))["pass"], False)

    def test_pure_java_with_a_native_library_line_fails(self) -> None:
        self.log.write_text(log_text("pure-java", native=True), encoding="utf-8")
        run = self.build()
        self.assertIsNotNone(run["nativeLibrary"])
        self.assertIs(run["pass"], False)

    def test_native_library_line_must_declare_the_rust_backend(self) -> None:
        bad = log_text("rust-ffm").replace("native-library backend=rust-ffm", "native-library backend=pure-java")
        self.assertIs(self.rust(log=bad)["pass"], False)

    def test_native_library_sha256_must_be_a_digest(self) -> None:
        bad = log_text("rust-ffm").replace(LIBRARY_SHA256, "not-a-digest")
        run = self.rust(log=bad)
        self.assertIsNone(run["nativeLibrary"])
        self.assertIs(run["pass"], False)

    def test_kernel_must_be_observably_injected_on_the_rust_arm(self) -> None:
        bad = log_text("rust-ffm").replace("injectedGroupedProjections=true", "injectedGroupedProjections=false")
        self.assertIs(self.rust(log=bad)["pass"], False)
        bad = log_text("rust-ffm").replace("matrixKernel=rust-ffm-quantized-v13", "matrixKernel=vector-api")
        self.assertIs(self.rust(log=bad)["pass"], False)

    def test_pure_java_must_not_report_an_injected_kernel(self) -> None:
        bad = log_text("pure-java").replace("injectedGroupedProjections=false", "injectedGroupedProjections=true")
        self.log.write_text(bad, encoding="utf-8")
        self.assertIs(self.build()["pass"], False)

    def test_any_present_cache_is_not_fresh(self) -> None:
        run = self.build(fresh_check="absent /root/.m2\npresent /root/.jbang\nabsent /root/.gradle\n")
        self.assertIs(run["freshMachine"], False)
        self.assertIs(run["pass"], False)

    def test_empty_fresh_check_is_not_fresh(self) -> None:
        self.assertIs(self.build(fresh_check="")["freshMachine"], False)

    def test_nonzero_exit_fails(self) -> None:
        self.assertIs(self.build(exit_code=1)["pass"], False)

    def test_wrong_java_fails(self) -> None:
        run = self.build(java_version_output='openjdk version "21.0.5" 2024-10-15\n')
        self.assertEqual(run["javaMajor"], 21)
        self.assertIs(run["pass"], False)

    def test_log_without_final_pass_line_fails(self) -> None:
        self.log.write_text(log_text("pure-java").replace("PASS cases=6 passed=6", "FAIL cases=6 passed=5"), encoding="utf-8")
        self.assertIs(self.build()["pass"], False)
        self.log.write_text(log_text("pure-java") + "later noise\n", encoding="utf-8")
        self.assertIs(self.build()["pass"], False)

    def test_time_must_advance(self) -> None:
        self.assertIs(self.build(completed_at="2026-09-17T10:00:00Z")["pass"], False)
        with self.assertRaises(ValueError):
            self.build(started_at="2026-09-17 10:00:00")

    def test_command_must_be_nonempty_strings(self) -> None:
        with self.assertRaises(ValueError):
            self.build(command=[])
        with self.assertRaises(ValueError):
            self.build(command=["jbang", ""])

    def test_java_major_parsing(self) -> None:
        self.assertEqual(writer.java_major('openjdk version "25" 2025-09-16\n'), 25)
        self.assertEqual(writer.java_major('openjdk version "25-ea" 2025-09-16\n'), 25)
        self.assertEqual(writer.java_major('java version "1.8.0_402"\n'), 1)
        self.assertIsNone(writer.java_major("garbage"))

    def test_passing_record_satisfies_the_component_report_validator(self) -> None:
        directory = "benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid"
        modules = self.dir / "validator"
        modules.mkdir()
        try:
            root = subprocess.run(
                ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            names = subprocess.run(
                ["git", "-C", root, "ls-tree", "--name-only", f"origin/main:{directory}"],
                check=True, capture_output=True, text=True,
            ).stdout.split()
            if "assemble_component_report.py" not in names:
                raise FileNotFoundError(directory)
            for name in names:  # the validator imports sibling helpers
                if name.endswith(".py"):
                    (modules / name).write_bytes(subprocess.run(
                        ["git", "-C", root, "show", f"origin/main:{directory}/{name}"],
                        check=True, capture_output=True,
                    ).stdout)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("origin/main assemble_component_report.py not reachable")
        sys.path.insert(0, str(modules))
        module_path = modules / "assemble_component_report.py"
        spec = importlib.util.spec_from_file_location("assemble_component_report", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for run in (self.build(), self.rust()):
            pinned = dict(run)
            pinned["outputLog"] = dict(run["outputLog"])
            pinned["outputLog"]["uri"] = run["outputLog"]["uri"].replace("<EVIDENCE_REVISION>", "a" * 40)
            module.validate_clean_host_run(pinned, "0.3.42")
        # The placeholder itself must not validate: the coordinator has to pin the revision.
        self.log.write_text(log_text("pure-java"), encoding="utf-8")
        with self.assertRaises(ValueError):
            module.validate_clean_host_run(self.build(), "0.3.42")
        with self.assertRaises(ValueError):
            failing = self.build(exit_code=1)
            failing["outputLog"]["uri"] = failing["outputLog"]["uri"].replace("<EVIDENCE_REVISION>", "a" * 40)
            module.validate_clean_host_run(failing, "0.3.42")


if __name__ == "__main__":
    unittest.main()
