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


class _FakeAdb:
  """Answers run_android's adb commands; records them in order.

  Attributes:
    commands: every `adb shell` command string, in order; a benchmark run is
      recorded as ("run", command).
  """

  def __init__(
      self, *, fail_run=None, rewritten="", mkdir_fails=False, gpu_loaded=True
  ):
    self.fail_run = fail_run
    self.rewritten = rewritten
    self.mkdir_fails = mkdir_fails
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
    lines = [
        f"INFO: [benchmark_litert_model.h:94] Model initialization: {n}0.00"
        " ms\n",
        "noise line\n",
    ]
    if n == 2 and self.gpu_loaded and "--use_gpu=true" in cmd[2]:
      lines.insert(
          0, "I0000 Initialized InferenceContext from serialized data.\n"
      )
    process = mock.MagicMock()
    process.stdout = iter(lines)
    process.returncode = 7 if n == self.fail_run else 0
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
        first,
        f"{_ROOT}/benchmark_model --graph={_ROOT}/m.tflite --use_gpu=true"
        f" {cache_args} --dry_run=true",
    )
    self.assertEqual(
        measured,
        f"{_ROOT}/benchmark_model --graph={_ROOT}/m.tflite --use_gpu=true"
        f" {cache_args} --report_peak_memory_footprint=true",
    )
    # mkdir, first process, sizes + mark, measured process, find, cleanup.
    steps = [c if isinstance(c, str) else c[0] for c in fake.commands]
    self.assertEqual(steps[-6], f"mkdir -p {cache}")
    self.assertEqual(steps[-5], "run")
    self.assertEqual(
        steps[-4],
        f"cd {cache} && wc -c * 2>/dev/null; touch {cache}.written",
    )
    self.assertEqual(steps[-3], "run")
    self.assertEqual(steps[-2], f"find {cache} -type f -newer {cache}.written")
    self.assertEqual(steps[-1], f"rm -rf {cache} {cache}.written")
    self.assertIn("Model initialization: 20.00 ms", result.output)
    self.assertNotIn("Model initialization: 10.00 ms", result.output)
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

  def test_a_failing_first_process_exits_1_and_removes_the_caches(self):
    fake = _FakeAdb(fail_run=1)
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 1)
    self.assertLen(fake.runs, 1)
    self.assertIn("Execution failed on device with exit code 7", result.output)
    self.assertIn("Error: Benchmark failed on device.\n", result.output)
    self.assertNotIn("Failed to execute benchmark on device", result.output)
    self.assertTrue(fake.commands[-1].startswith("rm -rf "))

  def test_a_failing_measured_process_exits_1_and_removes_the_caches(self):
    fake = _FakeAdb(fail_run=2)
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 1)
    self.assertLen(fake.runs, 2)
    self.assertTrue(fake.commands[-1].startswith("rm -rf "))
    self.assertNotIn("First process", result.output)

  def test_a_cache_directory_that_cannot_be_made_exits_1(self):
    fake = _FakeAdb(mkdir_fails=True)
    result = self._invoke(fake, "--cpu")
    self.assertEqual(result.exit_code, 1)
    self.assertEmpty(fake.runs)
    self.assertIn("mkdir: Permission denied", result.output)

  def test_npu_runs_once_without_the_cache_flags(self):
    fake = _FakeAdb()
    result = self._invoke(fake, "--npu")
    self.assertEqual(result.exit_code, 0, result.output)
    (run,) = fake.runs
    self.assertIn("--use_npu=true", run)
    self.assertNotIn("cache", run)
    self.assertFalse(any("benchmark_cache_" in str(c) for c in fake.commands))


if __name__ == "__main__":
  absltest.main()
