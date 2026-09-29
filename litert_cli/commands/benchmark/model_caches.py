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

"""benchmark_model's model caches and the two processes that use them.

A .tflite benchmark on the CPU or the GPU runs benchmark_model twice with the
same cache flags. The first process loads and compiles the model, writes
XNNPACK's weight cache and, on the GPU, the serialized GPU data into a cache
directory, and runs no inference. The second process is the measured one: it
can read those caches, and it reports the peak memory too.

The caches are an addition to the benchmark, not what it depends on. When a
process that carries the cache flags exits non-zero, or the cache directory
cannot be made, benchmark_model runs once without the cache flags, as it did
before the caches, and the CLI fails only if that run fails. Setting
LITERT_DISABLE_MODEL_CACHES=1 runs it once from the start. `report` says which
of these happened and, after the two processes, what the first one wrote and
whether the measured one used it, from the cache files and the lines both
processes logged.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re

from litert_cli.core import constants

# The first process only loads and compiles the model.
WARMUP_ARGS = ("--dry_run=true",)
PEAK_MEMORY_ARG = "--report_peak_memory_footprint=true"
# What the report says when LITERT_DISABLE_MODEL_CACHES=1 turned the two
# processes off.
DISABLED_LINE = (
    f"Model caches: off ({constants.ENV_LITERT_DISABLE_MODEL_CACHES}=1):"
    " benchmark_model ran once without the cache flags"
)
# Lines of a failed process shown before the run without the cache flags.
OUTPUT_TAIL_LINES = 20

_INIT = re.compile(r"Model initialization:\s+([\d.]+) ms")
# benchmark_model warns about the flags it does not know and goes on; a
# release without a cache flag logs them here (2.2.0 has no GPU
# serialization flags).
_FLAGS_IGNORED = re.compile(r"Unconsumed cmdline flags:\s*(.*\S)")
_GPU_SERIALIZATION_FLAG = "--gpu_serialization_dir"
# A flag it knows but cannot parse ends the process after this line.
_FLAG_NOT_PARSED = "Failed to parse flag"
_GPU_CACHE_LOADED = "Initialized InferenceContext from serialized data"
_GPU_CACHE_NOT_SAVED = "Failed to save serialized data"
# Some builds log it; the report never reads its absence as a miss.
_XNNPACK_CACHE_LOADED = "XNNPack weight cache loaded from"


def disabled() -> bool:
  """Whether LITERT_DISABLE_MODEL_CACHES=1 turns the two processes off.

  Read when a benchmark starts, not when the module loads.
  """
  return os.environ.get(constants.ENV_LITERT_DISABLE_MODEL_CACHES, "0") == "1"


def uses_caches(accelerator: str) -> bool:
  """Whether the benchmark runs the two processes (NPU runs once, as before)."""
  return accelerator in ("cpu", "gpu") and not disabled()


def cache_args(accelerator: str, cache_dir: str, model_name: str) -> list[str]:
  """The flags that write the caches into cache_dir, or read them from there.

  XNNPACK's weight cache holds the ops that run on the CPU, on the GPU too for
  the ops the GPU does not take. The GPU serializes its data only when it has
  both a directory and a key.

  Args:
    accelerator: cpu or gpu.
    cache_dir: The cache directory, as the binary sees it.
    model_name: The model's file name; its stem names the caches.
  """
  stem = pathlib.PurePosixPath(model_name).stem
  args = [f"--xnnpack_weight_cache_file_path={cache_dir}/{stem}.xnnpack_cache"]
  if accelerator == "gpu":
    args += [
        f"--gpu_serialization_dir={cache_dir}",
        f"--gpu_model_cache_key={stem}",
    ]
  return args


@dataclasses.dataclass
class Outcome:
  """What the processes of one benchmark did; `report` reads it.

  Attributes:
    first: The first process's output lines; empty when it did not run.
    measured: The measured process's output lines; empty when it did not run.
    written: Size in bytes of each cache file after the first process; None when
      the files could not be listed.
    rewritten: The cache files the measured process wrote again; None when that
      could not be checked.
    fallback: Why benchmark_model then ran once without the cache flags, or None
      when the measured process's results stand.
  """

  first: list[str] = dataclasses.field(default_factory=list)
  measured: list[str] = dataclasses.field(default_factory=list)
  written: dict[str, int] | None = None
  rewritten: list[str] | None = None
  fallback: str | None = None


def report(accelerator: str, outcome: Outcome) -> list[str]:
  """Text lines, one fact each: the mode first, then what the processes did.

  Args:
    accelerator: cpu or gpu.
    outcome: What the processes did.
  """
  if outcome.fallback is not None:
    lines = [
        f"Model caches: off ({outcome.fallback}): benchmark_model ran once"
        " without the cache flags, so the results above are a cold start"
        " with no peak memory"
    ]
    processes = (("first", outcome.first), ("measured", outcome.measured))
    for name, output in processes:
      for line in output:
        match = _FLAGS_IGNORED.search(line)
        start = match.start() if match else line.find(_FLAG_NOT_PARSED)
        if start >= 0:
          lines.append(f"The {name} process logged: {line[start:].strip()}")
          break
    return lines
  init = next((m.group(1) for m in map(_INIT.search, outcome.first) if m), None)
  lines = [
      "Model caches: on",
      "First process (compiles the model and writes the caches, no"
      f" inference): init {init} ms"
      if init
      else "First process: no 'Model initialization' line in its output",
  ]
  if outcome.written is None:
    lines.append("Could not list the cache files it wrote")
  else:
    files = [
        f"{name} ({size} B)" for name, size in sorted(outcome.written.items())
    ]
    lines.append(f"Caches it wrote: {', '.join(files) or 'none'}")
    if outcome.rewritten is None:
      lines.append(
          "Could not check whether the measured process wrote them again"
      )
    elif outcome.rewritten:
      lines.append(
          "The measured process wrote again:"
          f" {', '.join(sorted(outcome.rewritten))}"
      )
    elif outcome.written:
      lines.append("The measured process left them unchanged")
  lines += [
      "The measured process logged: "
      + line[line.index(_XNNPACK_CACHE_LOADED) :].strip()
      for line in outcome.measured
      if _XNNPACK_CACHE_LOADED in line
  ][:1]
  ignored: list[str] = []
  for output in (outcome.first, outcome.measured):
    for line in output:
      match = _FLAGS_IGNORED.search(line)
      if match and match.group(1) not in ignored:
        ignored.append(match.group(1))
  lines += [
      f"Flags this benchmark_model ignored (logged as unconsumed): {flags}"
      for flags in ignored
  ]
  if accelerator == "gpu":
    lines += [
        line.strip() for line in outcome.first if _GPU_CACHE_NOT_SAVED in line
    ]
    if any(_GPU_SERIALIZATION_FLAG in flags for flags in ignored):
      lines.append(
          f"This benchmark_model has no {_GPU_SERIALIZATION_FLAG}: neither"
          " process wrote or read a GPU cache"
      )
    elif any(_GPU_CACHE_LOADED in line for line in outcome.measured):
      lines.append(f"The measured process logged: {_GPU_CACHE_LOADED}")
    elif outcome.measured:
      lines.append(f"The measured process did not log: {_GPU_CACHE_LOADED}")
  if not outcome.measured:
    lines.append("No output lines of the measured process")
  return lines
