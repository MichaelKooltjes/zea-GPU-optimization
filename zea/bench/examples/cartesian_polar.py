import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["ZEA_DISABLE_CACHE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import ops
from zea.beamform.delays import compute_t0_delays_focused
from zea.beamform.phantoms import fish
from zea.probes import Probe
from zea.scan import Scan
from zea.visualize import pad_or_crop_extent, set_mpl_style
from zea.internal.core import DEFAULT_DYNAMIC_RANGE

zea.init_device(verbose=False)
set_mpl_style()

n_el = 128
aperture = 30e-3
probe_geometry = np.stack(
    [
        np.linspace(-aperture / 2, aperture / 2, n_el),
        np.zeros(n_el),
        np.zeros(n_el),
    ],
    axis=1,
)

probe = Probe(
    probe_geometry=probe_geometry,
    center_frequency=2.5e6,
    sampling_frequency=10e6,
)

sound_speed = 1540.0
n_tx = 8

tx_apodizations = np.ones((n_tx, probe.n_el)) * np.hanning(probe.n_el)[None]
angles = np.linspace(30, -30, n_tx) * np.pi / 180
focus_distances = np.ones(n_tx) * 15e-3
transmit_origins = np.zeros((n_tx, 3))
t0_delays = compute_t0_delays_focused(
    transmit_origins=transmit_origins,
    focus_distances=focus_distances,
    probe_geometry=probe.probe_geometry,
    polar_angles=angles,
    sound_speed=sound_speed,
)

scan = Scan(
    n_el=n_el,
    center_frequency=probe.center_frequency,
    sampling_frequency=probe.sampling_frequency,
    probe_geometry=probe.probe_geometry,
    t0_delays=t0_delays,
    tx_apodizations=tx_apodizations,
    focus_distances=focus_distances,
    transmit_origins=transmit_origins, #again, doesnt exist
    polar_angles=angles,
    initial_times=np.ones(n_tx) * 1e-6,
    n_ax=1024,
    lens_sound_speed=1000,
    lens_thickness=1e-3,
    sound_speed=sound_speed,
    xlims=(-20e-3, 20e-3),
    zlims=(0, 35e-3),
    n_tx=n_tx,
    n_ch=1,
)

# Initialize the fish phantom
scat_positions = fish()
n_scat = len(scat_positions)
simulation_parameters = dict(
    scatterer_positions=scat_positions.astype(np.float32),
    scatterer_magnitudes=np.ones(n_scat, dtype=np.float32),
)

pipeline = ops.Pipeline.from_default(beamformer="das", with_batch_dim=False)
pipeline.prepend(ops.Simulate())
pipeline.append(ops.Normalize(input_range=DEFAULT_DYNAMIC_RANGE, output_range=(0, 255)))

# Prepare parameters for the pipeline
scan.grid_type = "cartesian"  # cartesian grid is the default
parameters = pipeline.prepare_parameters(probe, scan)

# Run the pipeline
image_cart = pipeline(**parameters, **simulation_parameters)["data"]

# Define the extent for the cartesian grid
extent_cart = scan.extent.copy()

# Append ScanConvert to the pipeline
pipeline_sc = pipeline.copy()
pipeline_sc.append(ops.ScanConvert(order=3))
pipeline_sc.append(zea.ops.keras_ops.Clip(x_min=0, x_max=255))

# Prepare parameters for the pipeline
scan.grid_type = "polar"  # update grid type to polar
parameters = pipeline_sc.prepare_parameters(probe, scan)

image_polar = pipeline_sc(**parameters, **simulation_parameters)["data"]

# Define the extent for the polar grid
extent_polar = scan.extent.copy()

# Make sure the polar image has the same extent as the cartesian image
image_polar_corrected = pad_or_crop_extent(image_polar, extent_polar, extent_cart)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image_cart, cmap="gray", extent=extent_cart, vmin=0, vmax=255)
axs[0].set_xlabel("X (mm)")
axs[0].set_ylabel("Z (mm)")
axs[0].set_title("Cartesian")
axs[0].locator_params(nbins=4)

axs[1].imshow(image_polar_corrected, cmap="gray", extent=extent_cart, vmin=0, vmax=255)
axs[1].set_title("Polar")
axs[1].set_xlabel("X (mm)")
axs[1].set_ylabel("Z (mm)")
axs[1].locator_params(nbins=4)

plt.tight_layout()
plt.savefig("polar_grid_fish.png")
plt.close()