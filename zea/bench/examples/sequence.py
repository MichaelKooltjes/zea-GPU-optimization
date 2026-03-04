import os

os.environ["KERAS_BACKEND"] = "jax"
import keras

import zea
from zea import init_device, load_file
from zea.visualize import set_mpl_style
from zea.internal.notebooks import animate_images

n_frames = 15
n_tx = 11
n_tx_total = 127

init_device(verbose=False)
set_mpl_style()

pipeline = zea.Pipeline.from_default(enable_pfield=False, with_batch_dim=False)

# this can take a while to download
file_path = "hf://zeahub/zea-carotid-2023/2_cross_bifur_right_0000.hdf5"  # ~25GB
# so let's use the smaller one by default:
file_path = "hf://zeahub/zea-carotid-2023/2_cross_bifur_right_0000_small.hdf5"  # ~2.5GB

frames = list(range(n_frames))  # use first 15 frames for demonstration
data, scan, probe = load_file(file_path, "raw_data", indices=frames)

scan.set_transmits(n_tx)  # reduce number of transmits for faster processing
scan.zlims = (0, 0.04)  # reduce z-limits a bit for better visualizations
scan.xlims = probe.xlims
scan.n_ch = data.shape[-1]  # rf data

config = zea.Config(dynamic_range=(-40, 0))

images = []
n_frames = data.shape[0]
progbar = keras.utils.Progbar(n_frames, stateful_metrics=["frame"])

params = pipeline.prepare_parameters(probe, scan, config)

for frame_no in range(n_frames):
    output = pipeline(data=data[frame_no, scan.selected_transmits], **params)
    image = output.pop("data")
    params = output
    images.append(image)
    progbar.update(frame_no + 1)

animate_images(images, "./bmode_sequence.gif", scan, interval=100, cmap="gray")

images = []
n_frames = len(frames)
progbar = keras.utils.Progbar(n_frames, stateful_metrics=["frame"])

for idx, frame_no in enumerate(frames):
    tx_idx = int(round(idx * (n_tx_total - 1) / (n_frames - 1)))
    scan.set_transmits([tx_idx])
    raw_data_frame = data[frame_no, tx_idx][None, ...]
    with zea.log.set_level("WARNING"):  # to surpress info messages
        params = pipeline.prepare_parameters(probe, scan)
        output = pipeline(data=raw_data_frame, **params)
    images.append(output["data"])
    progbar.update(idx + 1)

animate_images(images, "./tx_sweep.gif", scan, interval=100, cmap="gray")