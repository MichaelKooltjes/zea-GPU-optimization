import os
import time
import json
import argparse
import platform
from statistics import mean, stdev

os.environ["KERAS_BACKEND"] = "torch"
os.environ["ZEA_DISABLE_CACHE"] = "0"
os.environ["ZEA_LOG_LEVEL"] = "INFO"


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def get_env_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "KERAS_BACKEND": os.environ.get("KERAS_BACKEND"),
        "ZEA_DISABLE_CACHE": os.environ.get("ZEA_DISABLE_CACHE"),
        "ZEA_LOG_LEVEL": os.environ.get("ZEA_LOG_LEVEL"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_error"] = str(e)
    return info


def reset_peak_mem():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_peak_mem():
    out = {}
    try:
        import torch
        if torch.cuda.is_available():
            out["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            out["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    except Exception:
        pass
    return out


def bench_pipeline(pipeline, data, params, warmup, iters, frames_per_call, return_numpy=False):
    for _ in range(warmup):
        cuda_sync()
        _ = pipeline(data=data, **params, return_numpy=return_numpy) if return_numpy else pipeline(data=data, **params)
        cuda_sync()

    run_times = []
    out = None
    reset_peak_mem()
    for _ in range(iters):
        cuda_sync()
        t0 = time.perf_counter()
        out = pipeline(data=data, **params, return_numpy=return_numpy) if return_numpy else pipeline(data=data, **params)
        cuda_sync()
        t1 = time.perf_counter()
        run_times.append(t1 - t0)

    stats = {
        "run_times_s": run_times,
        "run_time_mean_s": mean(run_times),
        "run_time_min_s": min(run_times),
        "run_time_max_s": max(run_times),
        "run_time_stdev_s": stdev(run_times) if len(run_times) >= 2 else 0.0,
        "frames_per_call": frames_per_call,
        "fps_mean": (frames_per_call / mean(run_times)) if mean(run_times) > 0 else None,
    }
    stats.update(get_peak_mem())

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
    parser.add_argument("--n_frames", type=int, default=12)
    parser.add_argument("--n_transmits", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--bmode_num_patches", type=int, default=1000)
    parser.add_argument("--doppler_num_patches", type=int, default=50)
    parser.add_argument("--out", default="bench_results_doppler.json")
    args = parser.parse_args()

    import zea
    import numpy as np
    from zea import init_device

    init_device(verbose=False)

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

    # ----------------------------------------
    # 2 | B-mode pipeline
    pipeline_bmode = zea.Pipeline.from_default(num_patches=args.bmode_num_patches, with_batch_dim=True)

    t2 = time.perf_counter()
    params_bmode = pipeline_bmode.prepare_parameters(probe, scan)
    t3 = time.perf_counter()
    results["bmode_prepare_time_s"] = t3 - t2

    frames_per_call = args.n_frames
    results["bmode"] = bench_pipeline(
        pipeline=pipeline_bmode,
        data=data,
        params=params_bmode,
        warmup=args.warmup,
        iters=args.iters,
        frames_per_call=frames_per_call,
        return_numpy=True,
    )
    # ----------------------------------------

    # ----------------------------------------
    # 3 | Doppler pipeline 
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
    results["doppler_prepare_time_s"] = t5 - t4

    results["doppler_prep"] = bench_pipeline(
        pipeline=pipeline_dop,
        data=data,
        params=params_dop,
        warmup=args.warmup,
        iters=args.iters,
        frames_per_call=frames_per_call,
        return_numpy=False,
    )
    # ----------------------------------------

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Doppler benchmark summary ===")
    print(f"Load time:            {results['load_time_s']:.2f} s")
    print(f"B-mode prepare time:  {results['bmode_prepare_time_s']:.2f} s")
    print(f"B-mode run mean:      {results['bmode']['run_time_mean_s']:.2f} s")
    print(f"B-mode FPS mean:      {results['bmode']['fps_mean']:.2f}")
    if "peak_allocated_bytes" in results["bmode"]:
        print(f"B-mode peak VRAM:     {results['bmode']['peak_allocated_bytes']/1024/1024:.1f} MiB")

    print(f"Doppler prepare time: {results['doppler_prepare_time_s']:.2f} s")
    print(f"Doppler run mean:     {results['doppler_prep']['run_time_mean_s']:.2f} s")
    print(f"Doppler FPS mean:     {results['doppler_prep']['fps_mean']:.2f}")
    if "peak_allocated_bytes" in results["doppler_prep"]:
        print(f"Doppler peak VRAM:    {results['doppler_prep']['peak_allocated_bytes']/1024/1024:.1f} MiB")

    print(f"Saved JSON:           {args.out}")


if __name__ == "__main__":
    main()
