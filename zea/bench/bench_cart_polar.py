import os
import time
import json
import argparse
import platform
from pathlib import Path
from statistics import mean, stdev

os.environ["KERAS_BACKEND"] = "jax"
os.environ["ZEA_DISABLE_CACHE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def jax_sync(x=None):
    try:
        if x is None:
            return
        try:
            x.block_until_ready()
        except Exception:
            pass
    except Exception:
        pass


def block_until_ready(out):
    if isinstance(out, dict):
        x = out.get("data", None)
        if x is not None:
            jax_sync(x)
    else:
        jax_sync(out)


def get_env_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "KERAS_BACKEND": os.environ.get("KERAS_BACKEND"),
        "ZEA_DISABLE_CACHE": os.environ.get("ZEA_DISABLE_CACHE"),
    }
    try:
        import jax
        info["jax"] = jax.__version__
        info["jax_platform"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as e:
        info["jax_error"] = str(e)
    return info


def timed_runs(cart_pipeline, cart_kwargs, polar_pipeline, polar_kwargs, warmup, iters, trace_dir=None):
    import jax.profiler

    out_cart = None
    out_polar = None
    for _ in range(warmup):
        out_cart = cart_pipeline(**cart_kwargs)
        block_until_ready(out_cart)
        out_polar = polar_pipeline(**polar_kwargs)
        block_until_ready(out_polar)

    run_times = []
    trace_cm = None
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        trace_cm = jax.profiler.trace(trace_dir, create_perfetto_link=False)
        trace_cm.__enter__()

    try:
        for i in range(iters):
            with jax.profiler.StepTraceAnnotation("cartesian+p o l a r", step_num=i):
                ts = time.perf_counter()

                out_cart = cart_pipeline(**cart_kwargs)
                block_until_ready(out_cart)

                out_polar = polar_pipeline(**polar_kwargs)
                block_until_ready(out_polar)

                te = time.perf_counter()
                run_times.append(te - ts)
    finally:
        if trace_cm:
            trace_cm.__exit__(None, None, None)

    stats = {
        "run_times_s": run_times,
        "run_time_mean_s": mean(run_times),
        "run_time_min_s": min(run_times),
        "run_time_max_s": max(run_times),
        "run_time_stdev_s": stdev(run_times) if len(run_times) >= 2 else 0.0,
    }

    if isinstance(out_cart, dict):
        stats["cart_output_keys"] = list(out_cart.keys())
        if "data" in out_cart:
            try:
                stats["cart_output_data_shape"] = list(out_cart["data"].shape)
            except Exception:
                stats["cart_output_data_shape"] = "unknown"

    if isinstance(out_polar, dict):
        stats["polar_output_keys"] = list(out_polar.keys())
        if "data" in out_polar:
            try:
                stats["polar_output_data_shape"] = list(out_polar["data"].shape)
            except Exception:
                stats["polar_output_data_shape"] = "unknown"

    return stats


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark for cartesian+polar scanconvert example")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--n_el", type=int, default=128)
    parser.add_argument("--aperture", type=float, default=30e-3)
    parser.add_argument("--center_frequency", type=float, default=2.5e6)
    parser.add_argument("--sampling_frequency", type=float, default=10e6)
    parser.add_argument("--sound_speed", type=float, default=1540.0)
    parser.add_argument("--n_tx", type=int, default=8)
    parser.add_argument("--n_ax", type=int, default=1024)
    parser.add_argument("--xlims", default="-0.02,0.02")
    parser.add_argument("--zlims", default="0.0,0.035")
    parser.add_argument("--lens_sound_speed", type=float, default=1000.0)
    parser.add_argument("--lens_thickness", type=float, default=1e-3)
    parser.add_argument("--beamformer", default="das")
    parser.add_argument("--scanconvert_order", type=int, default=3)
    parser.add_argument("--out", default="bench_results_cartesian_polar.json")
    parser.add_argument("--trace_dir", default="/data/jax_traces/cartesian_polar")
    args = parser.parse_args()

    import numpy as np
    import zea
    from zea import ops
    from zea.beamform.delays import compute_t0_delays_focused
    from zea.beamform.phantoms import fish
    from zea.internal.core import DEFAULT_DYNAMIC_RANGE
    from zea.probes import Probe
    from zea.scan import Scan

    zea.init_device(verbose=True)

    x0, x1 = [float(x.strip()) for x in args.xlims.split(",")]
    z0, z1 = [float(z.strip()) for z in args.zlims.split(",")]

    results = {
        "case": "cartesian_polar_scanconvert",
        "warmup": args.warmup,
        "iters": args.iters,
        "n_el": args.n_el,
        "aperture": args.aperture,
        "center_frequency": args.center_frequency,
        "sampling_frequency": args.sampling_frequency,
        "sound_speed": args.sound_speed,
        "n_tx": args.n_tx,
        "n_ax": args.n_ax,
        "xlims": [x0, x1],
        "zlims": [z0, z1],
        "lens_sound_speed": args.lens_sound_speed,
        "lens_thickness": args.lens_thickness,
        "beamformer": args.beamformer,
        "scanconvert_order": args.scanconvert_order,
        "env": get_env_info(),
        "trace_dir": args.trace_dir,
    }

    t0 = time.perf_counter()

    probe_geometry = np.stack(
        [
            np.linspace(-args.aperture / 2, args.aperture / 2, args.n_el),
            np.zeros(args.n_el),
            np.zeros(args.n_el),
        ],
        axis=1,
    )

    probe = Probe(
        probe_geometry=probe_geometry,
        center_frequency=args.center_frequency,
        sampling_frequency=args.sampling_frequency,
    )

    angles = np.linspace(30, -30, args.n_tx) * np.pi / 180
    focus_distances = np.ones(args.n_tx) * 15e-3
    transmit_origins = np.zeros((args.n_tx, 3))
    tx_apodizations = np.ones((args.n_tx, probe.n_el)) * np.hanning(probe.n_el)[None]

    t0_delays = compute_t0_delays_focused(
        transmit_origins=transmit_origins,
        focus_distances=focus_distances,
        probe_geometry=probe.probe_geometry,
        polar_angles=angles,
        sound_speed=args.sound_speed,
    )

    scan_cart = Scan(
        n_el=args.n_el,
        center_frequency=probe.center_frequency,
        sampling_frequency=probe.sampling_frequency,
        probe_geometry=probe.probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        focus_distances=focus_distances,
        transmit_origins=transmit_origins,
        polar_angles=angles,
        initial_times=np.ones(args.n_tx) * 1e-6,
        n_ax=args.n_ax,
        lens_sound_speed=args.lens_sound_speed,
        lens_thickness=args.lens_thickness,
        sound_speed=args.sound_speed,
        xlims=(x0, x1),
        zlims=(z0, z1),
        n_tx=args.n_tx,
        n_ch=1,
    )
    scan_cart.grid_type = "cartesian"

    scan_polar = Scan(
        n_el=args.n_el,
        center_frequency=probe.center_frequency,
        sampling_frequency=probe.sampling_frequency,
        probe_geometry=probe.probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        focus_distances=focus_distances,
        transmit_origins=transmit_origins,
        polar_angles=angles,
        initial_times=np.ones(args.n_tx) * 1e-6,
        n_ax=args.n_ax,
        lens_sound_speed=args.lens_sound_speed,
        lens_thickness=args.lens_thickness,
        sound_speed=args.sound_speed,
        xlims=(x0, x1),
        zlims=(z0, z1),
        n_tx=args.n_tx,
        n_ch=1,
    )
    scan_polar.grid_type = "polar"

    scat_positions = fish()
    n_scat = len(scat_positions)
    simulation_parameters = dict(
        scatterer_positions=scat_positions.astype(np.float32),
        scatterer_magnitudes=np.ones(n_scat, dtype=np.float32),
    )

    pipeline = ops.Pipeline.from_default(beamformer=args.beamformer, with_batch_dim=False)
    pipeline.prepend(ops.Simulate())
    pipeline.append(ops.Normalize(input_range=DEFAULT_DYNAMIC_RANGE, output_range=(0, 255)))

    pipeline_sc = pipeline.copy()
    pipeline_sc.append(ops.ScanConvert(order=args.scanconvert_order))
    pipeline_sc.append(zea.ops.keras_ops.Clip(x_min=0, x_max=255))

    t1 = time.perf_counter()
    results["setup_time_s"] = t1 - t0
    results["n_scat"] = int(n_scat)

    t2 = time.perf_counter()
    params_cart = pipeline.prepare_parameters(probe, scan_cart)
    t3 = time.perf_counter()
    results["cartesian_parameters_time_s"] = t3 - t2

    t4 = time.perf_counter()
    params_polar = pipeline_sc.prepare_parameters(probe, scan_polar)
    t5 = time.perf_counter()
    results["polar_parameters_time_s"] = t5 - t4

    trace_dir = f"{args.trace_dir}" if args.trace_dir else None
    stats = timed_runs(
        cart_pipeline=pipeline,
        cart_kwargs={**params_cart, **simulation_parameters},
        polar_pipeline=pipeline_sc,
        polar_kwargs={**params_polar, **simulation_parameters},
        warmup=args.warmup,
        iters=args.iters,
        trace_dir=trace_dir,
    )

    stats["trace_dir"] = trace_dir
    stats["tasks_per_call"] = 1
    stats["fps_mean"] = 1.0 / stats["run_time_mean_s"]
    results["cartesian_and_polar"] = stats

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Benchmark summary ===")
    print(f"Setup time:         {results['setup_time_s']:.2f} s")
    print(f"Cartesian prepare:  {results['cartesian_parameters_time_s']:.2f} s")
    print(f"Polar prepare:      {results['polar_parameters_time_s']:.2f} s")
    print(f"Run mean:           {results['cartesian_and_polar']['run_time_mean_s']:.2f} s  (min {results['cartesian_and_polar']['run_time_min_s']:.2f} s)")
    print(f"Task rate mean:     {results['cartesian_and_polar']['fps_mean']:.2f} (task = cartesian+polar)")
    if results["cartesian_and_polar"]["trace_dir"]:
        print(f"Trace dir:          {results['cartesian_and_polar']['trace_dir']}")
    print(f"Saved JSON:         {args.out}")


if __name__ == "__main__":
    main()
