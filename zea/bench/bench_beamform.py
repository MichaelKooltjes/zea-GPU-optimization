import os
import time

os.environ["KERAS_BACKEND"] = "jax"
os.environ["ZEA_DISABLE_CACHE"] = "0"
os.environ["ZEA_LOG_LEVEL"] = "INFO"

import jax
import jax.profiler
import numpy as np
from keras import ops
from pathlib import Path

import zea
from zea import init_device
from zea.beamform.tiling import compute_l2_num_patches, query_gpu_l2_cache_bytes
from zea.internal.core import DataTypes
from zea.ops import (
    TOFCorrectionLoopReorder,
    ReshapeGrid,
    Pipeline,
)
from zea.ops.pipeline import PatchedGrid

# ─────────────────────────────────────────────────────────────────────────────
# CHOOSE OPTIMIZATION MODE
# ─────────────────────────────────────────────────────────────────────────────
#
#   "baseline"     – original unfused TOFCorrection + DelayAndSum
#   "fused"        – fused kernel, never materialises n_tx*n_pix*n_el*n_ch
#   "l2_aware"     – fused + num_patches chosen to fit GPU L2 cache
#   "loop_reorder" – element-major inner loop for better gather locality
#   "float16"      – fused pipeline with input data cast to float16
#
MODE = "float16"

# Patch count used by baseline / fused / loop_reorder / float16 modes.
# l2_aware ignores this and computes its own value at runtime.
NUM_PATCHES = 200

N_WARMUP = 1
N_RUNS = 3
TRACE_DIR = "/data/jax_traces/z_beamform_profiling"
# ─────────────────────────────────────────────────────────────────────────────

init_device(verbose=False)

with zea.File("hf://zeahub/zea-rotating-disk/L115V_1radsec.hdf5") as file:
    scan = file.scan()
    data = file.load_data("raw_data")
    probe = file.probe()

n_frames, n_tx, n_ax, n_el, n_ch = data.shape   # n_ch=1 for raw RF, 2 for IQ
n_pix = int(np.prod(scan.grid.shape[:-1]))

# ── Build the pipeline for the selected mode ──────────────────────────────────

if MODE == "baseline":
    beamform = zea.ops.Beamform(
        beamformer="delay_and_sum",
        num_patches=NUM_PATCHES,
        fused=False,
    )
    input_data = data
    num_patches = NUM_PATCHES

elif MODE == "fused":
    beamform = zea.ops.Beamform(
        beamformer="delay_and_sum",
        num_patches=NUM_PATCHES,
        fused=True,
    )
    input_data = data
    num_patches = NUM_PATCHES

elif MODE == "l2_aware":
    l2_bytes = query_gpu_l2_cache_bytes()
    # one transmit at a time in registers, not the full n_tx batch at once.
    num_patches = compute_l2_num_patches(
        n_pix, n_tx=n_tx, n_el=n_el, n_ch=n_ch,
        l2_cache_bytes=l2_bytes, fused=True,
    )
    print(f"L2 cache: {l2_bytes // 1024 // 1024} MB  →  num_patches = {num_patches}")
    beamform = zea.ops.Beamform(
        beamformer="delay_and_sum",
        num_patches=num_patches,
        fused=True,
    )
    input_data = data

elif MODE == "loop_reorder":
    reshape = ReshapeGrid()
    reshape.output_data_type = DataTypes.BEAMFORMED_DATA
    beamform = Pipeline([
        PatchedGrid([TOFCorrectionLoopReorder()], num_patches=NUM_PATCHES),
        reshape,
    ])
    input_data = data
    num_patches = NUM_PATCHES

elif MODE == "float16":
    beamform = zea.ops.Beamform(
        beamformer="delay_and_sum",
        num_patches=NUM_PATCHES,
        fused=True,
    )
    input_data = ops.cast(data, "float16")
    num_patches = NUM_PATCHES

else:
    raise ValueError(
        f"Unknown MODE: {MODE!r}. "
        "Choose from: baseline, fused, l2_aware, loop_reorder, float16"
    )

print(f"Mode: {MODE}  |  n_pix={n_pix}  n_ch={n_ch}  num_patches={num_patches}")

# ── Prepare parameters ────────────────────────────────────────────────────────

params = beamform.prepare_parameters(probe, scan)

# ── Warmup ────────────────────────────────────────────────────────────────────

for _ in range(N_WARMUP):
    out = beamform(data=input_data, **params)
    try:
        out["data"].block_until_ready()
    except Exception:
        pass

# ── Timed runs with JAX profiler trace ───────────────────────────────────────

Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)
times = []

with jax.profiler.trace(TRACE_DIR):
    for i in range(N_RUNS):
        start = time.perf_counter()

        with jax.profiler.StepTraceAnnotation("beamform", step_num=i):
            out = beamform(data=input_data, **params)
            try:
                out["data"].block_until_ready()
            except Exception:
                pass

        times.append(time.perf_counter() - start)

# ── Results ───────────────────────────────────────────────────────────────────

mean_runtime = np.mean(times)
fps = 1.0 / mean_runtime

print(f"Trace saved to: {TRACE_DIR}")
print(f"Mean runtime:   {mean_runtime:.6f} s")
print(f"Amount of runs: {N_RUNS}")
print(f"Frames per second generated: {fps:.2f}")
