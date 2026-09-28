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
# lines benchmark_model prints. Exits 7 on call N when fail<N> exists. Writes
# the cache files its flags name on the first call, and on the second too when
# rewrite exists; logs the GPU cache load on the second call, or warns that it
# ignores the GPU flags when nogpuflags exists. Like benchmark_model, it keeps
# the first copy of a repeated flag.
_FAKE = """#!/bin/sh
dir=$(dirname "$0")
printf '%s\\n' "$@" -- >> "$dir/calls.txt"
n=$(grep -c -- '^--$' "$dir/calls.txt")
[ -f "$dir/fail$n" ] && exit 7
for arg in "$@"; do
  case "$arg" in
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

  def _invoke(self, *extra_args: str) -> testing.Result:
    with (
        mock.patch.object(
            desktop, "_ensure_desktop_binary", return_value=self.fake
        ),
        mock.patch.object(constants, "DEFAULT_QUIET", True),
        mock.patch.object(benchmark_cli.utils, "enable_quiet_mode"),
    ):
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
    self.assertIn(
        "This benchmark_model has no --gpu_serialization_dir: neither process"
        " wrote or read a GPU cache",
        result.output,
    )

  def test_a_failing_first_process_exits_1_before_the_measured_one(self):
    (self.dir / "fail1").touch()
    result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 1)
    self.assertLen(self._calls(), 1)
    self.assertIn("Execution failed on desktop with exit code 7", result.output)
    self.assertIn("Error: Benchmark failed on desktop.", result.output)

  def test_a_failing_measured_process_exits_1(self):
    (self.dir / "fail2").touch()
    result = self._invoke("--cpu")
    self.assertEqual(result.exit_code, 1)
    self.assertLen(self._calls(), 2)
    self.assertNotIn("First process", result.output)

  def test_npu_runs_once_without_the_cache_flags(self):
    result = self._invoke("--npu")
    self.assertEqual(result.exit_code, 0, result.output)
    (call,) = self._calls()
    self.assertEqual(
        call, [f"--graph={self.model.resolve()}", "--use_npu=true"]
    )
    self.assertNotIn("First process", result.output)


if __name__ == "__main__":
  absltest.main()
