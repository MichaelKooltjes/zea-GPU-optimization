import os
import time

os.environ["KERAS_BACKEND"] = "jax"
os.environ["ZEA_DISABLE_CACHE"] = "0"
os.environ["ZEA_LOG_LEVEL"] = "INFO"

import zea
import numpy as np
from zea import init_device
import jax.profiler
from pathlib import Path

# n_frames = 25
# n_transmits = 10

init_device(verbose=False)

with zea.File("hf://zeahub/zea-rotating-disk/L115V_1radsec.hdf5") as file:
    scan = file.scan()

    # selected_tx = np.linspace(0, scan.n_tx_total - 1, n_transmits, dtype=int)
    # selected_frames = slice(n_frames)
    # scan.set_transmits(selected_tx)

    data = file.load_data("raw_data")
    probe = file.probe()

# beamform 
beamform = zea.ops.Beamform(
    beamformer="delay_and_sum",
    num_patches=200,
)

params = beamform.prepare_parameters(probe, scan)

trace_dir = "/data/jax_traces/beamform_only"
Path(trace_dir).mkdir(parents=True, exist_ok=True)

# warmup
for _ in range(1):
    out = beamform(data=data, **params)
    try:
        out["data"].block_until_ready()
    except:
        pass

# tensorboard trace
n_runs = 3
times = []
with jax.profiler.trace(trace_dir):
    for i in range(n_runs):
        start = time.time()

        with jax.profiler.StepTraceAnnotation("beamform", step_num=i):
            out = beamform(data=data, **params)
            try:
                out["data"].block_until_ready()
            except:
                pass

        end = time.time()
        times.append(end - start)

print("Trace saved to:", trace_dir)
mean_runtime = np.mean(times)
fps = 1 / mean_runtime

print(f"Mean runtime: {mean_runtime:.6f} s")
print(f"Amount of runs: {len(times)}")
print(f"Frames per second generated: {fps:.2f}")
