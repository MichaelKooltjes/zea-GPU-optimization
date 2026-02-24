import os
import time
import json
import argparse
import platform
from statistics import mean, stdev

os.environ["KERAS_BACKEND"] = "torch"
os.environ["ZEA_DISABLE_CACHE"] = "0"

#benchmark part
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

def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark for 3D grid beamforming pipeline")
    parser.add_argument("--path", default="hf://zeahub/phantoms/2025_12_16_cirs_focused_3d.hdf5")
    parser.add_argument("--indices", default="0", help="Comma-separated indices, e.g. 0 or 0,1,2")
    parser.add_argument("--chunks", type=int, default=1024/2)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--out", default="bench_results_3d_beamforming.json")
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

    
    scan.n_ch = 2 # IQ data, should be stored in file but isn't currently
    scan.zlims = (0, 25e-3) # reduce z-limits a bit for better visualization

    #hasattr because it gave me errors for a nonexistent scan.grid_size_y
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
    )
    # ----------------------------------------


    # ----------------------------------------
    # 3 | Parameters
    t2 = time.perf_counter()
    parameters = pipeline.prepare_parameters(probe, scan)
    t3 = time.perf_counter()
    results["parameters_time_s"] = t3 - t2

    #resetting GPU peak memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    # ----------------------------------------


    # ----------------------------------------
    # 4 | Warm-up
    for _ in range(args.warmup):
        cuda_sync()
        _ = pipeline(data=rf_data, **parameters)
        cuda_sync()
    # ----------------------------------------


    # ----------------------------------------
    # 5 | Runs
    run_times = []
    out = None
    for _ in range(args.iters):
        cuda_sync()
        ts = time.perf_counter()
        out = pipeline(data=rf_data, **parameters)
        cuda_sync()
        te = time.perf_counter()
        run_times.append(te - ts)

    results["run_times_s"] = run_times
    results["run_time_mean_s"] = mean(run_times)
    results["run_time_min_s"] = min(run_times)
    results["run_time_max_s"] = max(run_times)
    results["run_time_stdev_s"] = stdev(run_times) if len(run_times) >= 2 else 0.0

    frames_per_call = max(1, len(indices))
    results["frames_per_call"] = frames_per_call
    results["fps_mean"] = frames_per_call / results["run_time_mean_s"]

    #peak VRAM
    try:
        import torch
        if torch.cuda.is_available():
            results["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass

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
    # 6 | Results
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    #readable results
    print("\n=== Benchmark summary ===")
    print(f"Load time:    {results['load_time_s']:.2f} s")
    print(f"Prepare time: {results['parameters_time_s']:.2f} s")
    print(f"Run mean:     {results['run_time_mean_s']:.2f} s  (min {results['run_time_min_s']:.2f} s)")
    print(f"FPS mean:     {results['fps_mean']:.2f} (frames_per_call={results['frames_per_call']})")

    if "peak_vram_bytes" in results:
        print(f"Peak VRAM:    {results['peak_vram_bytes']/1024/1024:.1f} MiB")
    print(f"Saved JSON:   {args.out}")

if __name__ == "__main__":
    main()
