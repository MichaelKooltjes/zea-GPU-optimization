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


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark for sequence example")
    parser.add_argument("--path", default="hf://zeahub/zea-carotid-2023/2_cross_bifur_right_0000_small.hdf5")
    parser.add_argument("--n_frames", type=int, default=15)
    parser.add_argument("--n_tx", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--out", default="bench_results_sequence.json")
    parser.add_argument("--trace_dir", default="/data/jax_traces/sequence")
    args = parser.parse_args()

    import zea
    from zea import init_device, load_file

    init_device(verbose=True)

    results = {
        "case": "sequence",
        "path": args.path,
        "n_frames": args.n_frames,
        "n_tx": args.n_tx,
        "warmup": args.warmup,
        "iters": args.iters,
        "env": get_env_info(),
        "trace_dir": args.trace_dir,
    }

    # ----------------------------------------
    # 1 | Load
    t0 = time.perf_counter()
    frames = list(range(args.n_frames))
    data, scan, probe = load_file(args.path, "raw_data", indices=frames)
    t1 = time.perf_counter()

    results["load_time_s"] = t1 - t0
    results["data_shape"] = list(getattr(data, "shape", []))
    # ----------------------------------------

    # ----------------------------------------
    # 2 | Pipeline + scan settings
    pipeline = zea.Pipeline.from_default(enable_pfield=False, with_batch_dim=False)

    scan.set_transmits(args.n_tx)
    scan.zlims = (0, 0.04)
    scan.xlims = probe.xlims
    scan.n_ch = data.shape[-1]
    # ----------------------------------------

    # ----------------------------------------
    # 3 | Prepare parameters
    t2 = time.perf_counter()
    params = pipeline.prepare_parameters(probe, scan)
    t3 = time.perf_counter()

    params = dict(params)
    params.pop("data", None)

    results["parameters_time_s"] = t3 - t2
    # ----------------------------------------

    # ----------------------------------------
    # 4 | Warm-up
    out_last = None
    for _ in range(args.warmup):
        for frame_no in range(data.shape[0]):
            out_last = pipeline(data=data[frame_no, scan.selected_transmits], **params)
            block_until_ready(out_last)
    # ----------------------------------------

    # ----------------------------------------
    # 5 | Runs
    import jax.profiler

    run_times = []
    trace_cm = None
    if args.trace_dir:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        trace_cm = jax.profiler.trace(args.trace_dir, create_perfetto_link=False)
        trace_cm.__enter__()

    try:
        for i in range(args.iters):
            with jax.profiler.StepTraceAnnotation("sequence", step_num=i):
                ts = time.perf_counter()

                for frame_no in range(data.shape[0]):
                    out_last = pipeline(data=data[frame_no, scan.selected_transmits], **params)
                    block_until_ready(out_last)

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

    results["frames_per_call"] = int(data.shape[0])
    results["fps_mean"] = results["frames_per_call"] / results["run_time_mean_s"]

    if isinstance(out_last, dict):
        results["output_keys"] = list(out_last.keys())
        if "data" in out_last:
            try:
                results["output_data_shape"] = list(out_last["data"].shape)
            except Exception:
                results["output_data_shape"] = "unknown"
    else:
        results["output_type"] = str(type(out_last))
    # ----------------------------------------

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Benchmark summary ===")
    print(f"Load time:     {results['load_time_s']:.2f} s")
    print(f"Prepare time:  {results['parameters_time_s']:.2f} s")
    print(f"Run mean:      {results['run_time_mean_s']:.2f} s  (min {results['run_time_min_s']:.2f} s)")
    print(f"FPS mean:      {results['fps_mean']:.2f} (frames_per_call={results['frames_per_call']})")
    if args.trace_dir:
        print(f"Trace dir:     {args.trace_dir}")
    print(f"Saved JSON:    {args.out}")


if __name__ == "__main__":
    main()
