import os
import time
import json
import argparse
import platform
from pathlib import Path
from statistics import mean, stdev

os.environ["KERAS_BACKEND"] = "jax"
os.environ["ZEA_DISABLE_CACHE"] = "0"


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


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark for 3D grid beamforming pipeline")
    parser.add_argument("--path", default="hf://zeahub/phantoms/2025_12_16_cirs_focused_3d.hdf5")
    parser.add_argument("--indices", default="0", help="Comma-separated indices, e.g. 0 or 0,1,2")
    parser.add_argument("--chunks", type=int, default=1024/2)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--out", default="bench_results_3d_beamforming.json")
    parser.add_argument("--trace_dir", default="/data/jax_traces/3D_beam")
    args = parser.parse_args()

    from zea import init_device
    from zea.data import load_file
    from zea.ops import (
        Pipeline,
        Demodulate,
        Map,
        EnvelopeDetect,
        ReshapeGrid,
        Normalize,
        LogCompress,
        TOFCorrection,
        DelayAndSum,
    )

    init_device(verbose=True)

    #parsing indices
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip() != ""]

    results = {
        "case": "3d_beamforming",
        "path": args.path,
        "indices": indices,
        "chunks": args.chunks,
        "downscale": args.downscale,
        "warmup": args.warmup,
        "iters": args.iters,
        "env": get_env_info(),
        "trace_dir": args.trace_dir,
    }

    # ----------------------------------------
    # 1 | Load
    t0 = time.perf_counter()
    rf_data, scan, probe = load_file(
        path=args.path,
        indices=indices,
        data_type="raw_data",
    )
    t1 = time.perf_counter()

    results["load_time_s"] = t1 - t0
    results["rf_shape"] = list(getattr(rf_data, "shape", []))
    # ----------------------------------------

    scan.n_ch = 2
    scan.zlims = (0, 25e-3)

    if hasattr(scan, "grid_size_x"):
        scan.grid_size_x //= args.downscale
    if hasattr(scan, "grid_size_y"):
        scan.grid_size_y //= args.downscale
    if hasattr(scan, "grid_size_z"):
        scan.grid_size_z //= args.downscale

    results["grid_shape"] = list(getattr(scan, "grid", []).shape) if getattr(scan, "grid", None) is not None else None

    # ----------------------------------------
    # 2 | Pipeline
    pipeline = Pipeline(
        [
            Demodulate(),
            Map(
                [TOFCorrection(), DelayAndSum()],
                argnames="flatgrid",
                chunks=args.chunks,
            ),
            ReshapeGrid(),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
        with_batch_dim=True,
        jit_options="pipeline",
    )
    # ----------------------------------------

    # ----------------------------------------
    # 3 | Parameters
    t2 = time.perf_counter()
    parameters = pipeline.prepare_parameters(probe, scan)
    t3 = time.perf_counter()
    results["parameters_time_s"] = t3 - t2
    # ----------------------------------------

    # ----------------------------------------
    # 4 | Warm-up
    out = None
    for _ in range(args.warmup):
        out = pipeline(data=rf_data, **parameters)
        block_until_ready(out)
    # ----------------------------------------

    # ----------------------------------------
    # 5 | Runs
    run_times = []

    trace_cm = None
    if args.trace_dir:
        import jax.profiler
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        trace_cm = jax.profiler.trace(args.trace_dir, create_perfetto_link=False)
        trace_cm.__enter__()

    try:
        for i in range(args.iters):
            with jax.profiler.StepTraceAnnotation("step", step_num=i):
                ts = time.perf_counter()
                out = pipeline(data=rf_data, **parameters)
                block_until_ready(out)
                te = time.perf_counter()
                run_times.append(te - ts)
    finally:
        if trace_cm:
            trace_cm.__exit__(None, None, None)

    results["run_times_s"] = run_times
    results["run_time_mean_s"] = mean(run_times)
    results["run_time_min_s"] = min(run_times)
    results["run_time_max_s"] = max(run_times)
    results["run_time_stdev_s"] = stdev(run_times) if len(run_times) >= 2 else 0.0

    frames_per_call = max(1, len(indices))
    results["frames_per_call"] = frames_per_call
    results["fps_mean"] = frames_per_call / results["run_time_mean_s"]

    #sanity check
    if isinstance(out, dict):
        results["output_keys"] = list(out.keys())
        if "data" in out:
            try:
                results["output_data_shape"] = list(out["data"].shape)
            except Exception:
                results["output_data_shape"] = "unknown"
    else:
        results["output_type"] = str(type(out))
    # ----------------------------------------

    # ----------------------------------------
    # 6 | Results
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    #readable results
    print("\n=== Benchmark summary ===")
    print(f"Load time:    {results['load_time_s']:.2f} s")
    print(f"Prepare time: {results['parameters_time_s']:.2f} s")
    print(f"Run mean:     {results['run_time_mean_s']:.2f} s  (min {results['run_time_min_s']:.2f} s)")
    print(f"FPS mean:     {results['fps_mean']:.2f} (frames_per_call={results['frames_per_call']})")
    if args.trace_dir:
        print(f"Trace dir:    {args.trace_dir}")
    print(f"Saved JSON:   {args.out}")


if __name__ == "__main__":
    main()
