import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["ZEA_DISABLE_CACHE"] = "0"
os.environ["ZEA_LOG_LEVEL"] = "INFO"

import zea
import numpy as np
from zea import init_device
import jax.profiler
from pathlib import Path

n_frames = 10
n_transmits = 10

init_device(verbose=False)

with zea.File("hf://zeahub/zea-rotating-disk/L115V_1radsec.hdf5") as file:
    scan = file.scan()

    # selected_tx = np.linspace(0, scan.n_tx_total - 1, n_transmits, dtype=int)
    # selected_frames = slice(n_frames)
    # scan.set_transmits(selected_tx)

    data = file.load_data("raw_data")
    probe = file.probe()

# ---- Beamform op only ----
beamform = zea.ops.Beamform(
    beamformer="delay_and_sum",
    num_patches=10,
)

params = beamform.prepare_parameters(probe, scan)

trace_dir = "/data/jax_traces/beamform_only"
Path(trace_dir).mkdir(parents=True, exist_ok=True)

# warmup (important for JIT)
for _ in range(1):
    out = beamform(data=data, **params)
    try:
        out["data"].block_until_ready()
    except:
        pass

# tensorboard trace
with jax.profiler.trace(trace_dir):
    for i in range(5):
        with jax.profiler.StepTraceAnnotation("beamform", step_num=i):
            out = beamform(data=data, **params)
            try:
                out["data"].block_until_ready()
            except:
                pass

print("Trace saved to:", trace_dir)
