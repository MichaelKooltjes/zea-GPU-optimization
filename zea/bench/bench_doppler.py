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


def timed_runs(pipeline, data, params, warmup, iters, trace_dir=None, return_numpy=False):
    out = None
    for _ in range(warmup):
        out = pipeline(data=data, **params, return_numpy=return_numpy) if return_numpy else pipeline(data=data, **params)
        block_until_ready(out)

    run_times = []
    trace_cm = None
    if trace_dir:
        import jax.profiler
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        trace_cm = jax.profiler.trace(trace_dir, create_perfetto_link=False)
        trace_cm.__enter__()

    try:
        for i in range(iters):
            with jax.profiler.StepTraceAnnotation("step", step_num=i):
                ts = time.perf_counter()
                out = pipeline(data=data, **params, return_numpy=return_numpy) if return_numpy else pipeline(data=data, **params)
                block_until_ready(out)
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

    if isinstance(out, dict):
        stats["output_keys"] = list(out.keys())
        if "data" in out:
            try:
                stats["output_data_shape"] = list(out["data"].shape)
            except Exception:
                stats["output_data_shape"] = "unknown"
    else:
        stats["output_type"] = str(type(out))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark for zea doppler example pipelines")
    parser.add_argument("--path", default="hf://zeahub/zea-rotating-disk/L115V_1radsec.hdf5")
    parser.add_argument("--n_frames", type=int, default=25)
    parser.add_argument("--n_transmits", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--bmode_num_patches", type=int, default=1000)
    parser.add_argument("--doppler_num_patches", type=int, default=20)
    parser.add_argument("--out", default="bench_results_doppler.json")
    parser.add_argument("--trace_dir", default="/data/jax_traces/doppler")
    args = parser.parse_args()

    import zea
    import numpy as np
    from zea import init_device

    init_device(verbose=True)

    results = {
        "case": "doppler_example",
        "path": args.path,
        "n_frames": args.n_frames,
        "n_transmits": args.n_transmits,
        "warmup": args.warmup,
        "iters": args.iters,
        "bmode_num_patches": args.bmode_num_patches,
        "doppler_num_patches": args.doppler_num_patches,
        "env": get_env_info(),
        "trace_dir": args.trace_dir,
    }

    # ----------------------------------------
    # 1 | Load
    t0 = time.perf_counter()
    with zea.File(args.path) as file:
        scan = file.scan()

        selected_tx = np.linspace(0, scan.n_tx_total - 1, args.n_transmits, dtype=int)
        selected_frames = slice(args.n_frames)
        scan.set_transmits(selected_tx)

        data = file.load_data("raw_data", indices=(selected_frames, selected_tx))
        probe = file.probe()
    t1 = time.perf_counter()

    results["load_time_s"] = t1 - t0
    results["data_shape"] = list(getattr(data, "shape", []))
    results["n_tx_total"] = getattr(scan, "n_tx_total", None)
    # ----------------------------------------

    frames_per_call = max(1, args.n_frames)
    results["frames_per_call"] = frames_per_call

    # ----------------------------------------
    # 2 | B-mode pipeline [outside of current scope]
    # pipeline_bmode = zea.Pipeline.from_default(num_patches=args.bmode_num_patches, with_batch_dim=True)

    # t2 = time.perf_counter()
    # params_bmode = pipeline_bmode.prepare_parameters(probe, scan)
    # t3 = time.perf_counter()
    # results["bmode_parameters_time_s"] = t3 - t2

    # bmode_trace = f"{args.trace_dir}/bmode" if args.trace_dir else None
    # results["bmode"] = timed_runs(
    #     pipeline=pipeline_bmode,
    #     data=data,
    #     params=params_bmode,
    #     warmup=args.warmup,
    #     iters=args.iters,
    #     trace_dir=bmode_trace,
    #     return_numpy=True,
    # )

    # results["bmode"]["fps_mean"] = frames_per_call / results["bmode"]["run_time_mean_s"]
    # results["bmode"]["trace_dir"] = bmode_trace
    # ----------------------------------------

    # ----------------------------------------
    # 3 | Doppler prep pipeline
    pipeline_dop = zea.Pipeline(
        [
            zea.ops.Demodulate(),
            zea.ops.Beamform(beamformer="delay_and_sum", num_patches=args.doppler_num_patches),
            zea.ops.ChannelsToComplex(),
        ],
        jit_options="pipeline",
    )

    t4 = time.perf_counter()
    params_dop = pipeline_dop.prepare_parameters(probe, scan)
    t5 = time.perf_counter()
    results["doppler_parameters_time_s"] = t5 - t4

    dop_trace = f"{args.trace_dir}" if args.trace_dir else None
    results["doppler_prep"] = timed_runs(
        pipeline=pipeline_dop,
        data=data,
        params=params_dop,
        warmup=args.warmup,
        iters=args.iters,
        trace_dir=dop_trace,
        return_numpy=False,
    )

    results["doppler_prep"]["fps_mean"] = frames_per_call / results["doppler_prep"]["run_time_mean_s"]
    results["doppler_prep"]["trace_dir"] = dop_trace
    # ----------------------------------------

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Benchmark summary ===")
    print(f"Load time:           {results['load_time_s']:.2f} s")
    # print(f"B-mode prepare time: {results['bmode_parameters_time_s']:.2f} s")
    # print(f"B-mode run mean:     {results['bmode']['run_time_mean_s']:.2f} s  (min {results['bmode']['run_time_min_s']:.2f} s)")
    # print(f"B-mode FPS mean:     {results['bmode']['fps_mean']:.2f} (frames_per_call={frames_per_call})")
    # if results["bmode"]["trace_dir"]:
    #     print(f"B-mode trace dir:    {results['bmode']['trace_dir']}")

    print(f"Doppler prep time:   {results['doppler_parameters_time_s']:.2f} s")
    print(f"Doppler run mean:    {results['doppler_prep']['run_time_mean_s']:.2f} s  (min {results['doppler_prep']['run_time_min_s']:.2f} s)")
    print(f"Doppler FPS mean:    {results['doppler_prep']['fps_mean']:.2f} (frames_per_call={frames_per_call})")
    if results["doppler_prep"]["trace_dir"]:
        print(f"Doppler trace dir:   {results['doppler_prep']['trace_dir']}")

    print(f"Saved JSON:          {args.out}")


if __name__ == "__main__":
    main()