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

"""Tests for the desktop benchmark target, with a fake benchmark_model."""

import os
import pathlib
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
from click import testing
from litert_cli.commands.benchmark import cli as benchmark_cli
from litert_cli.commands.benchmark import desktop
from litert_cli.core import constants

# Logs its arguments to calls.txt, one call per "--" line, and prints the
# lines benchmark_model prints. Exits 7 on call N when fail<N> exists, and
# exits 1 with benchmark_model's parse error on a call with --dry_run=true
# when reject exists; warns that it ignores --dry_run=true when nodryrun
# exists. Writes the cache files its flags name on the first call, and on the
# second too when rewrite exists; logs the GPU cache load on the second call,
# or warns that it ignores the GPU flags when nogpuflags exists. Like
# benchmark_model, it keeps the first copy of a repeated flag.
_FAKE = """#!/bin/sh
dir=$(dirname "$0")
printf '%s\\n' "$@" -- >> "$dir/calls.txt"
n=$(grep -c -- '^--$' "$dir/calls.txt")
echo "INFO: STARTING!"
[ -f "$dir/fail$n" ] && exit 7
for arg in "$@"; do
  case "$arg" in
    --dry_run=true)
      if [ -f "$dir/reject" ]; then
        echo "ERROR: Failed to parse flag 'dry_run' against argv '$arg'"
        exit 1
      fi
      [ -f "$dir/nodryrun" ] && echo "WARN: Unconsumed cmdline flags: $arg"
      ;;
    --xnnpack_weight_cache_file_path=*) [ -n "$x" ] || x="${arg#*=}" ;;
    --gpu_serialization_dir=*) [ -n "$g" ] || g="${arg#*=}" ;;
    --gpu_model_cache_key=*) [ -n "$k" ] || k="${arg#*=}" ;;
  esac
done
if [ -n "$g" ] && [ -f "$dir/nogpuflags" ]; then
  echo "WARN: Unconsumed cmdline flags:" \
    "--gpu_serialization_dir=$g --gpu_model_cache_key=$k"
  g=""
fi
if [ "$n" = 1 ] || [ -f "$dir/rewrite" ]; then
  [ -n "$x" ] && printf x > "$x"
  [ -n "$g" ] && printf gg > "$g/${k}_mldrift_program_cache.bin"
fi
[ "$n" = 2 ] && [ -n "$g" ] && echo "I0000 delegate_kernel.cc:866]" \
  "Initialized InferenceContext from serialized data."
echo "INFO: [benchmark_litert_model.h:94] Model initialization: 1${n}0.00 ms"
echo "noise line"
exit 0
"""


class RunDesktopTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # tempfile rather than absltest's create_tempdir(): the latter reads an
    # absl flag, which is not parsed when the file runs under pytest.
    self.dir = pathlib.Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
    self.fake = self.dir / "benchmark_model"
    self.fake.write_text(_FAKE)
    self.fake.chmod(0o755)
    self.model = self.dir / "m.tflite"
    self.model.write_bytes(b"\0")
    self.plain = [f"--graph={self.model.resolve()}"]

  def _invoke(self, *extra_args: str, env=None) -> testing.Result:
    with (
        mock.patch.object(
            desktop, "_ensure_desktop_binary", return_value=self.fake
        ),
        mock.patch.object(constants, "DEFAULT_QUIET", True),
        mock.patch.object(benchmark_cli.utils, "enable_quiet_mode"),
        mock.patch.dict(os.environ, env or {}),
    ):
      os.environ.pop(constants.ENV_LITERT_DISABLE_MODEL_CACHES, None)
      os.environ.update(env or {})
      return testing.CliRunner().invoke(
          benchmark_cli.benchmark_cmd,
          [str(self.model), "--desktop", *extra_args],
      )

  def _calls(self) -> list[list[str]]:
    text = (self.dir / "calls.txt").read_text()
    return [call.split("\n")[:-1] for call in text.split("--\n")[:-1]]

  def test_the_first_process_writes_the_caches_the_measured_one_reads(self):
    result = self._invoke("--gpu")
    self.assertEqual(result.exit_code, 0, result.output)
    first, measured = self._calls()
    base = [f"--graph={self.model.resolve()}", "--use_gpu=true"]
    self.assertEqual(first[:2], base)
    self.assertEqual(measured[:2], base)
    cache_args = first[2:-1]
    cache_dir = os.path.dirname(cache_args[0].split("=", 1)[1])
    self.assertEqual(
        cache_args,
        [
            f"--xnnpack_weight_cache_file_path={cache_dir}/m.xnnpack_cache",
            f"--gpu_serialization_dir={cache_dir}",
            "--gpu_model_cache_key=m",
        ],
    )
    self.assertEqual(first[-1], "--dry_run=true")
    self.assertEqual(
        measured[2:], cache_args + ["--report_peak_memory_footprint=true"]
    )
    self.assertFalse(os.path.exists(cache_dir))
    self.assertIn("Model initialization: 120.00 ms", result.output)
    self.assertNotIn("Model initialization: 110.00 ms", result.output)
    self.assertNotIn("noise line", result.output)
    self.assertIn("Model caches: on\n", result.output)
    self.assertIn(
        "First process (compiles the model and writes the caches, no"
        " inference): init 110.00 ms",
        result.output,
    )
    self.assertIn(
        "Caches it wrote: m.xnnpack_cache (1 B), m_mldrift_program_cache.bin"
        " (2 B)",
        result.output,
    )
    self.assertIn("The measured process left them unchanged", result.output)
    self.assertIn(
        "The measured process logged: Initialized InferenceContext from"
        " serialized data",
        result.output,
    )
    self.assertNotIn("without the cache flags", result.output)

  def test_cpu_writes_only_the_xnnpack_cache(self):
    result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    first, _ = self._calls()
    self.assertLen([a for a in first if a.startswith("--gpu_")], 0)
    self.assertIn("Caches it wrote: m.xnnpack_cache (1 B)\n", result.output)

  def test_a_cache_written_again_is_named(self):
    (self.dir / "rewrite").touch()
    result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn(
        "The measured process wrote again: m.xnnpack_cache", result.output
    )

  def test_a_binary_without_the_gpu_flags_is_named(self):
    (self.dir / "nogpuflags").touch()
    result = self._invoke("--gpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Caches it wrote: m.xnnpack_cache (1 B)\n", result.output)
    first, _ = self._calls()
    cache_dir = first[2].split("=", 1)[1].rsplit("/", 1)[0]
    self.assertIn(
        "Flags this benchmark_model ignored (logged as unconsumed):"
        f" --gpu_serialization_dir={cache_dir} --gpu_model_cache_key=m\n",
        result.output,
    )
    self.assertIn(
        "This benchmark_model has no --gpu_serialization_dir: neither process"
        " wrote or read a GPU cache",
        result.output,
    )

  def test_a_binary_that_ignores_dry_run_is_named(self):
    (self.dir / "nodryrun").touch()
    result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertLen(self._calls(), 2)
    self.assertIn("Model caches: on\n", result.output)
    self.assertIn(
        "Flags this benchmark_model ignored (logged as unconsumed):"
        " --dry_run=true\n",
        result.output,
    )

  def test_a_failing_first_process_runs_one_process_without_the_flags(self):
    (self.dir / "fail1").touch()
    result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    first, fallback = self._calls()
    self.assertIn("--dry_run=true", first)
    self.assertEqual(fallback, self.plain)
    self.assertIn("Execution failed on desktop with exit code 7", result.output)
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
    self.assertIn("Model initialization: 120.00 ms", result.output)
    self.assertIn(
        "Model caches: off (the first process exited with 7): benchmark_model"
        " ran once without the cache flags, so the results above are a cold"
        " start with no peak memory\n",
        result.output,
    )
    self.assertNotIn("First process", result.output)
    self.assertNotIn("Error:", result.output)

  def test_a_failing_measured_process_runs_one_process_without_the_flags(
      self,
  ):
    (self.dir / "fail2").touch()
    result = self._invoke("--gpu")
    self.assertEqual(result.exit_code, 0, result.output)
    first, measured, fallback = self._calls()
    self.assertIn("--dry_run=true", first)
    self.assertIn("--report_peak_memory_footprint=true", measured)
    self.assertEqual(fallback, self.plain + ["--use_gpu=true"])
    self.assertIn(
        "Model caches: off (the measured process exited with 7)",
        result.output,
    )
    self.assertIn("Model initialization: 130.00 ms", result.output)
    # The failed measured process's numbers are not on the screen.
    self.assertNotIn("Model initialization: 120.00 ms", result.output)

  def test_a_failing_measured_process_and_a_failing_fallback_exit_1(self):
    (self.dir / "fail2").touch()
    (self.dir / "fail3").touch()
    result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 1)
    self.assertLen(self._calls(), 3)
    self.assertIn("Error: Benchmark failed on desktop.", result.output)

  def test_a_cache_directory_that_cannot_be_made_runs_one_process(self):
    with mock.patch.object(
        desktop.tempfile,
        "TemporaryDirectory",
        side_effect=OSError(28, "No space left on device"),
    ):
      result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(self._calls(), [self.plain])
    self.assertIn(
        "Model caches: off (could not make a cache directory ([Errno 28] No"
        " space left on device))",
        result.output,
    )

  def test_a_binary_that_rejects_dry_run_is_quoted_in_the_report(self):
    (self.dir / "reject").touch()
    result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertLen(self._calls(), 2)
    self.assertIn(
        "Model caches: off (the first process exited with 1)", result.output
    )
    self.assertIn(
        "The first process logged: Failed to parse flag 'dry_run' against"
        " argv '--dry_run=true'\n",
        result.output,
    )

  def test_a_failing_process_without_the_flags_exits_1(self):
    (self.dir / "fail1").touch()
    (self.dir / "fail2").touch()
    result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 1)
    self.assertLen(self._calls(), 2)
    self.assertIn("Running benchmark_model once without", result.output)
    self.assertIn("Full output for debugging:", result.output)
    self.assertIn("Error: Benchmark failed on desktop.", result.output)
    self.assertNotIn("Model caches:", result.output)

  def test_the_environment_variable_runs_one_process_from_the_start(self):
    result = self._invoke(
        "--cpu", env={constants.ENV_LITERT_DISABLE_MODEL_CACHES: "1"}
    )
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(self._calls(), [self.plain])
    self.assertIn(
        "Model caches: off (LITERT_DISABLE_MODEL_CACHES=1): benchmark_model"
        " ran once without the cache flags\n",
        result.output,
    )
    self.assertNotIn("First process", result.output)
    # Only the value 1 turns the caches off.
    (self.dir / "calls.txt").unlink()
    result = self._invoke(
        "--cpu", env={constants.ENV_LITERT_DISABLE_MODEL_CACHES: "0"}
    )
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertLen(self._calls(), 2)
    self.assertIn("Model caches: on\n", result.output)

  def test_npu_runs_once_without_the_cache_flags(self):
    result = self._invoke("--npu")
    self.assertEqual(result.exit_code, 0, result.output)
    (call,) = self._calls()
    self.assertEqual(
        call, [f"--graph={self.model.resolve()}", "--use_npu=true"]
    )
    self.assertNotIn("First process", result.output)
    self.assertNotIn("Model caches", result.output)

  def test_a_cache_file_that_disappears_is_left_out_of_the_listing(self):
    (self.dir / "a.bin").write_bytes(b"aa")
    (self.dir / "b.bin").write_bytes(b"b")
    real_stat = pathlib.Path.stat

    def stat(path, *args, **kwargs):
      if path.name == "a.bin":
        raise FileNotFoundError(path)
      return real_stat(path, *args, **kwargs)

    with mock.patch.object(pathlib.Path, "stat", stat):
      files = desktop._cache_files(str(self.dir))
    self.assertEqual(
        {name: size for name, (size, _) in files.items()},
        {
            "b.bin": 1,
            "benchmark_model": len(_FAKE),
            "m.tflite": 1,
        },
    )


if __name__ == "__main__":
  absltest.main()
