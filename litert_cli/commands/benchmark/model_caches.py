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
can read those caches, and it reports the peak memory too. `report` says what
the first process wrote and whether the measured process used it, from the
cache files and the lines both processes logged.
"""

from __future__ import annotations

import pathlib
import re

# The first process only loads and compiles the model.
WARMUP_ARGS = ("--dry_run=true",)
PEAK_MEMORY_ARG = "--report_peak_memory_footprint=true"

_INIT = re.compile(r"Model initialization:\s+([\d.]+) ms")
# A binary without the GPU serialization flags (2.2.0) warns and ignores them.
_GPU_FLAGS_IGNORED = re.compile(
    r"Unconsumed cmdline flags:.*--gpu_serialization_dir"
)
_GPU_CACHE_LOADED = "Initialized InferenceContext from serialized data"
_GPU_CACHE_NOT_SAVED = "Failed to save serialized data"
# Some builds log it; the report never reads its absence as a miss.
_XNNPACK_CACHE_LOADED = "XNNPack weight cache loaded from"


def uses_caches(accelerator: str) -> bool:
  """Whether the benchmark runs the two processes (NPU runs once, as before)."""
  return accelerator in ("cpu", "gpu")


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


def report(
    accelerator: str,
    first_output: list[str],
    measured_output: list[str],
    written: dict[str, int],
    rewritten: list[str],
) -> list[str]:
  """What the first process wrote, and whether the measured process used it.

  Args:
    accelerator: cpu or gpu.
    first_output: The first process's output lines.
    measured_output: The measured process's output lines.
    written: Size in bytes of each cache file after the first process.
    rewritten: The cache files the measured process wrote again; None when
      that could not be checked.

  Returns:
    Text lines, one fact each.
  """
  init = next(
      (m.group(1) for m in map(_INIT.search, first_output) if m), None
  )
  lines = [
      "First process (compiles the model and writes the caches, no"
      f" inference): init {init} ms"
      if init
      else "First process: no 'Model initialization' line in its output"
  ]
  files = [f"{name} ({size} B)" for name, size in sorted(written.items())]
  lines.append(f"Caches it wrote: {', '.join(files) or 'none'}")
  if rewritten is None:
    lines.append(
        "Could not check whether the measured process wrote them again"
    )
  elif rewritten:
    lines.append(
        f"The measured process wrote again: {', '.join(sorted(rewritten))}"
    )
  elif written:
    lines.append("The measured process left them unchanged")
  lines += [
      "The measured process logged: "
      + line[line.index(_XNNPACK_CACHE_LOADED) :].strip()
      for line in measured_output
      if _XNNPACK_CACHE_LOADED in line
  ][:1]
  if accelerator == "gpu":
    lines += [
        line.strip() for line in first_output if _GPU_CACHE_NOT_SAVED in line
    ]
    if any(_GPU_FLAGS_IGNORED.search(line) for line in first_output):
      lines.append(
          "This benchmark_model has no --gpu_serialization_dir: neither"
          " process wrote or read a GPU cache"
      )
    elif any(_GPU_CACHE_LOADED in line for line in measured_output):
      lines.append(f"The measured process logged: {_GPU_CACHE_LOADED}")
    elif measured_output:
      lines.append(f"The measured process did not log: {_GPU_CACHE_LOADED}")
  if not measured_output:
    lines.append("No output lines of the measured process")
  return lines
