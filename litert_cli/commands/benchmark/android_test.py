# Copyright 2026 The LiteRT CLI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for the Android benchmark target (no device: adb is faked)."""

import pathlib
import shutil
import subprocess
import tempfile
from unittest import mock

from absl.testing import absltest
from click import testing
from litert_cli.commands.benchmark import android
from litert_cli.commands.benchmark import cli as benchmark_cli
from litert_cli.core import constants

_ROOT = constants.LITERT_CLI_ANDROID_ROOT
_PARSE_ERROR = (
    "ERROR: Failed to parse flag 'dry_run' against argv '--dry_run=true'"
)


class _FakeAdb:
  """Answers run_android's adb commands; records them in order.

  Attributes:
    commands: every `adb shell` command string, in order; a benchmark run is
      recorded as ("run", command).
    runs: every benchmark_model command line, in order.
  """

  def __init__(
      self,
      *,
      fail_runs=(),
      reject_dry_run=False,
      rewritten="",
      mkdir_fails=False,
      list_fails=False,
      gpu_loaded=True,
  ):
    self.fail_runs = fail_runs
    self.reject_dry_run = reject_dry_run
    self.rewritten = rewritten
    self.mkdir_fails = mkdir_fails
    self.list_fails = list_fails
    self.gpu_loaded = gpu_loaded
    self.commands = []
    self.runs = []

  def run(self, cmd, **kwargs):
    del kwargs
    command = " ".join(cmd[2:])
    self.commands.append(command)
    if self.mkdir_fails and command.startswith("mkdir -p ") and (
        "benchmark_cache_" in command
    ):
      return subprocess.CompletedProcess(cmd, 1, "mkdir: Permission denied\n")
    if "wc -c" in command:
      if self.list_fails:
        return subprocess.CompletedProcess(cmd, 1, "error: device offline\n")
      return subprocess.CompletedProcess(
          cmd,
          0,
          "     160 m.xnnpack_cache\n57764240 m_mldrift_program_cache.bin\n"
          "57764400 total\n",
      )
    if command.startswith("find "):
      return subprocess.CompletedProcess(cmd, 0, self.rewritten)
    return subprocess.CompletedProcess(cmd, 0, "")

  def popen(self, cmd, **kwargs):
    del kwargs
    self.runs.append(cmd[2])
    self.commands.append(("run", cmd[2]))
    n = len(self.runs)
    lines = ["INFO: STARTING!\n"]
    process = mock.MagicMock()
    process.returncode = 0
    if self.reject_dry_run and "--dry_run=true" in cmd[2]:
      lines.append(_PARSE_ERROR + "\n")
      process.returncode = 1
    elif n in self.fail_runs:
      process.returncode = 7
    else:
      if n == 2 and self.gpu_loaded and "--use_gpu=true" in cmd[2]:
        lines.append(
            "I0000 Initialized InferenceContext from serialized data.\n"
        )
      lines += [
          f"INFO: [benchmark_litert_model.h:94] Model initialization: {n}0.00"
          " ms\n",
          "noise line\n",
      ]
    process.stdout = iter(lines)
    return process


class RunAndroidTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # tempfile rather than absltest's create_tempdir(): the latter reads an
    # absl flag, which is not parsed when the file runs under pytest.
    self.dir = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
    self.model = self.dir / "m.tflite"
    self.model.write_bytes(b"\0")
    self.plain = f"{_ROOT}/benchmark_model --graph={_ROOT}/m.tflite"

  def _invoke(self, fake: _FakeAdb, *extra_args: str) -> testing.Result:
    with (
        mock.patch.object(android.subprocess, "run", side_effect=fake.run),
        mock.patch.object(android.subprocess, "Popen", side_effect=fake.popen),
        mock.patch.object(android.android_utils, "check_adb"),
        mock.patch.object(
            android.android_utils, "get_android_abi", return_value="arm64-v8a"
        ),
        mock.patch.object(
            android.android_utils,
            "find_android_binary",
            return_value=self.dir / "benchmark_model",
        ),
        mock.patch.object(android.android_utils, "push_file_to_device"),
        mock.patch.object(
            android.npu, "push_npu_runtime_libraries", return_value=_ROOT
        ),
        mock.patch.object(
            android.npu, "get_soc_target_model", return_value="sm8750"
        ),
        mock.patch.object(
            android.android_utils,
            "find_npu_dispatch_lib",
            return_value=pathlib.Path("libLiteRtDispatch_Qualcomm.so"),
        ),
        mock.patch.object(
            android.android_utils,
            "find_npu_compiler_plugin_lib",
            return_value=pathlib.Path("libLiteRtCompilerPlugin_Qualcomm.so"),
        ),
        mock.patch.object(constants, "DEFAULT_QUIET", True),
        mock.patch.object(benchmark_cli.utils, "enable_quiet_mode"),
    ):
      return testing.CliRunner().invoke(
          benchmark_cli.benchmark_cmd,
          [str(self.model), "--android", *extra_args],
      )

  def _cache_dir(self, fake: _FakeAdb) -> str:
    (mkdir,) = [
        c
        for c in fake.commands
        if isinstance(c, str) and "benchmark_cache_" in c
    ][:1]
    return mkdir.split()[-1]

  def _steps(self, fake: _FakeAdb) -> list[str]:
    return [c if isinstance(c, str) else c[0] for c in fake.commands]

  def test_the_first_process_writes_the_caches_the_measured_one_reads(self):
    fake = _FakeAdb()
    result = self._invoke(fake, "--gpu")
    self.assertEqual(result.exit_code, 0, result.output)
    cache = self._cache_dir(fake)
    self.assertRegex(cache, rf"^{_ROOT}/benchmark_cache_[0-9a-f]{{8}}$")
    first, measured = fake.runs
    cache_args = (
        f"--xnnpack_weight_cache_file_path={cache}/m.xnnpack_cache"
        f" --gpu_serialization_dir={cache} --gpu_model_cache_key=m"
    )
    self.assertEqual(
        first, f"{self.plain} --use_gpu=true {cache_args} --dry_run=true"
    )
    self.assertEqual(
        measured,
        f"{self.plain} --use_gpu=true {cache_args}"
        " --report_peak_memory_footprint=true",
    )
    # mkdir, first process, sizes + mark, measured process, find, cleanup.
    steps = self._steps(fake)
    self.assertEqual(steps[-6], f"mkdir -p {cache}")
    self.assertEqual(steps[-5], "run")
    self.assertEqual(
        steps[-4],
        f"cd {cache} && wc -c * 2>/dev/null; touch {cache}.written",
    )
    self.assertEqual(steps[-3], "run")
    self.assertEqual(
        steps[-2],
        f"find {cache} -type f -newer {cache}.written 2>/dev/null",
    )
    self.assertEqual(steps[-1], f"rm -rf {cache} {cache}.written")
    self.assertIn("Model initialization: 20.00 ms", result.output)
    self.assertNotIn("Model initialization: 10.00 ms", result.output)
    self.assertIn("Model caches: on\n", result.output)
    self.assertIn(
        "First process (compiles the model and writes the caches, no"
        " inference): init 10.00 ms",
        result.output,
    )
    self.assertIn(
        "Caches it wrote: m.xnnpack_cache (160 B), m_mldrift_program_cache.bin"
        " (57764240 B)\n",
        result.output,
    )
    self.assertIn("The measured process left them unchanged", result.output)
    self.assertIn(
        "The measured process logged: Initialized InferenceContext from"
        " serialized data",
        result.output,
    )
    self.assertNotIn("without the cache flags", result.output)

  def test_the_cache_flags_are_quoted_for_the_device_shell(self):
    self.model = self.dir / "my model.tflite"
    self.model.write_bytes(b"\0")
    fake = _FakeAdb()
    result = self._invoke(fake, "--gpu")
    self.assertEqual(result.exit_code, 0, result.output)
    cache = self._cache_dir(fake)
    first, _ = fake.runs
    self.assertIn(
        f"'--xnnpack_weight_cache_file_path={cache}/my model.xnnpack_cache'"
        f" --gpu_serialization_dir={cache} '--gpu_model_cache_key=my model'",
        first,
    )

  def test_a_cache_written_again_or_not_loaded_is_named(self):
    fake = _FakeAdb(
        rewritten=f"{_ROOT}/benchmark_cache_x/m.xnnpack_cache\n",
        gpu_loaded=False,
    )
    result = self._invoke(fake, "--gpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn(
        "The measured process wrote again: m.xnnpack_cache", result.output
    )
    self.assertIn(
        "The measured process did not log: Initialized InferenceContext from"
        " serialized data",
        result.output,
    )

  def test_a_failing_first_process_runs_one_process_without_the_flags(self):
    fake = _FakeAdb(fail_runs=(1,))
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    first, fallback = fake.runs
    self.assertIn("--dry_run=true", first)
    self.assertEqual(fallback, self.plain)
    # The caches are removed before the process without the flags runs.
    steps = self._steps(fake)
    self.assertEqual(steps[-3:], [
        "run",
        f"rm -rf {self._cache_dir(fake)} {self._cache_dir(fake)}.written",
        "run",
    ])
    self.assertIn("Execution failed on device with exit code 7", result.output)
    self.assertIn(
        "Last 1 lines of its output (LITERT_VERBOSE=1 shows all of it):\n"
        "INFO: STARTING!\n",
        result.output,
    )
    self.assertIn(
        "Running benchmark_model once without the cache flags: the first"
        " process exited with 7\n",
        result.output,
    )
    self.assertIn("Model initialization: 20.00 ms", result.output)
    self.assertIn(
        "Model caches: off (the first process exited with 7): benchmark_model"
        " ran once without the cache flags, so the results above are a cold"
        " start with no peak memory\n",
        result.output,
    )
    self.assertNotIn("Error:", result.output)

  def test_a_failing_measured_process_runs_one_process_without_the_flags(
      self,
  ):
    fake = _FakeAdb(fail_runs=(2,))
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertLen(fake.runs, 3)
    self.assertEqual(fake.runs[2], self.plain)
    self.assertIn(
        "Model caches: off (the measured process exited with 7)",
        result.output,
    )
    self.assertIn("Model initialization: 30.00 ms", result.output)
    self.assertNotIn("Model initialization: 20.00 ms", result.output)

  def test_a_failing_measured_process_and_a_failing_fallback_exit_1(self):
    fake = _FakeAdb(fail_runs=(2, 3))
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 1)
    self.assertLen(fake.runs, 3)
    self.assertIn("Error: Benchmark failed on device.\n", result.output)

  def test_a_binary_that_rejects_dry_run_is_quoted_in_the_report(self):
    fake = _FakeAdb(reject_dry_run=True)
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertLen(fake.runs, 2)
    self.assertIn(
        "Model caches: off (the first process exited with 1)", result.output
    )
    self.assertIn(
        "The first process logged: Failed to parse flag 'dry_run' against"
        " argv '--dry_run=true'\n",
        result.output,
    )

  def test_a_failing_process_without_the_flags_exits_1(self):
    fake = _FakeAdb(fail_runs=(1, 2))
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 1)
    self.assertLen(fake.runs, 2)
    self.assertIn("Full output for debugging:", result.output)
    self.assertIn("Error: Benchmark failed on device.\n", result.output)
    self.assertNotIn("Failed to execute benchmark on device", result.output)
    self.assertNotIn("Model caches:", result.output)

  def test_a_cache_directory_that_cannot_be_made_runs_one_process(self):
    fake = _FakeAdb(mkdir_fails=True)
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(fake.runs, [self.plain])
    cache = self._cache_dir(fake)
    self.assertIn(
        f"Model caches: off ('adb shell mkdir -p {cache}' exited with 1:"
        " mkdir: Permission denied)",
        result.output,
    )

  def test_a_cache_listing_that_fails_does_not_stop_the_run(self):
    fake = _FakeAdb(list_fails=True)
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertLen(fake.runs, 2)
    self.assertIn("Model caches: on\n", result.output)
    self.assertIn("Could not list the cache files it wrote\n", result.output)
    self.assertNotIn("Caches it wrote", result.output)

  def test_npu_runs_once_without_the_cache_flags(self):
    fake = _FakeAdb()
    result = self._invoke(fake, "--npu")
    self.assertEqual(result.exit_code, 0, result.output)
    (run,) = fake.runs
    self.assertIn("--use_npu=true", run)
    self.assertNotIn("cache", run)
    self.assertFalse(any("benchmark_cache_" in str(c) for c in fake.commands))
    self.assertNotIn("Model caches", result.output)


if __name__ == "__main__":
  absltest.main()
