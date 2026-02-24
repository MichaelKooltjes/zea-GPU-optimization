import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["ZEA_DISABLE_CACHE"] = "0"
os.environ["ZEA_LOG_LEVEL"] = "INFO"

import matplotlib.pyplot as plt

import zea
from zea.doppler import color_doppler
import numpy as np
from zea import init_device
from zea.visualize import set_mpl_style
from zea.internal.notebooks import animate_images

n_frames = 10
n_transmits = 10

init_device(verbose=False)

with zea.File("hf://zeahub/zea-rotating-disk/L115V_1radsec.hdf5") as file:
    scan = file.scan()

    # Let's use a little bit less data for this demo
    selected_tx = np.linspace(0, scan.n_tx_total - 1, n_transmits, dtype=int)
    selected_frames = slice(n_frames)
    scan.set_transmits(selected_tx)

    data = file.load_data("raw_data", indices=(selected_frames, selected_tx))
    probe = file.probe()
#B-mode
pipeline = zea.Pipeline.from_default(num_patches=1000, with_batch_dim=True)
params = pipeline.prepare_parameters(probe, scan)
bmode = pipeline(data=data, **params, return_numpy=True)["data"]

animate_images(bmode, "doppler.gif", scan)

#doppler
pipeline = zea.Pipeline(
    [
        zea.ops.Demodulate(),
        zea.ops.Beamform(beamformer="delay_and_sum", num_patches=100),
        zea.ops.ChannelsToComplex(),
    ],
    jit_options="pipeline",
)

params = pipeline.prepare_parameters(probe, scan)
output = pipeline(data=data, **params)
data4doppler = output["data"]

# pulse_repetition_frequency = 1 / sum(scan.time_to_next_transmit[0])
# d = color_doppler(
#     data4doppler,
#     probe.center_frequency,
#     pulse_repetition_frequency,
#     scan.sound_speed,
#     hamming_size=10,  # spatial smoothing with Hamming window
# )
# plt.imshow(d * 100, cmap="bwr", extent=scan.extent * 1e3)
# plt.title("Doppler image (cm/s)")
# plt.xlabel("X (mm)")
# plt.ylabel("Z (mm)")
# plt.colorbar()
# plt.show()