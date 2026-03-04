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
    parser = argparse.ArgumentParser(description="Baseline benchmark for working.py example")
    parser.add_argument(
        "--path",
        default="hf://zeahub/picmus/database/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_patches", type=int, default=100)
    parser.add_argument("--zmax", type=float, default=0.06)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--out", default="bench_results_working.json")
    parser.add_argument("--trace_dir", default="/data/jax_traces/working")
    args = parser.parse_args()

    import zea
    from zea.data import load_file
    from zea.ops import Pipeline, EnvelopeDetect, Normalize, LogCompress, Beamform

    zea.init_device(verbose=False)

    results = {
        "case": "working_example",
        "path": args.path,
        "index": args.index,
        "num_patches": args.num_patches,
        "zmax": args.zmax,
        "warmup": args.warmup,
        "iters": args.iters,
        "env": get_env_info(),
        "trace_dir": args.trace_dir,
    }

    # ----------------------------------------
    # 1 | Load
    t0 = time.perf_counter()
    data, scan, probe = load_file(
        path=args.path,
        indices=[args.index],
        data_type="raw_data",
    )
    t1 = time.perf_counter()

    results["load_time_s"] = t1 - t0
    results["data_shape"] = list(getattr(data, "shape", []))

    data_frame = data[0]
    # ----------------------------------------

    # ----------------------------------------
    # 2 | Scan settings
    scan.n_ch = 2
    scan.xlims = probe.xlims
    scan.zlims = (0, float(args.zmax))
    # ----------------------------------------

    # ----------------------------------------
    # 3 | Pipelines
    pipeline_default = Pipeline.from_default(
        num_patches=args.num_patches,
        baseband=True,
        enable_pfield=False,
        with_batch_dim=False,
        jit_options="pipeline",
    )

    pipeline_custom = Pipeline(
        operations=[
            Beamform(
                beamformer="das",
                enable_pfield=False,
                num_patches=args.num_patches,
            ),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
        with_batch_dim=False,
        jit_options="pipeline",
    )
    # ----------------------------------------

    # ----------------------------------------
    # 4 | Prepare parameters (default)
    t2 = time.perf_counter()
    params_default = pipeline_default.prepare_parameters(probe, scan)
    t3 = time.perf_counter()
    results["default_parameters_time_s"] = t3 - t2

    params_default = dict(params_default)
    params_default.pop("dynamic_range", None)
    params_default.pop("data", None)

    # prepare parameters (custom)
    t4 = time.perf_counter()
    params_custom = pipeline_custom.prepare_parameters(probe, scan)
    t5 = time.perf_counter()
    results["custom_parameters_time_s"] = t5 - t4

    params_custom = dict(params_custom)
    params_custom.pop("dynamic_range", None)
    params_custom.pop("data", None)
    # ----------------------------------------

    # ----------------------------------------
    # 5 | Task (one full run computes everything)
    inputs_single = {pipeline_default.key: data_frame}
    inputs_single_custom = {pipeline_custom.key: data_frame}

    def run_task():
        out_a = pipeline_default(**inputs_single, **params_default)
        block_until_ready(out_a)

        out_b = pipeline_custom(**inputs_single_custom, **params_custom, dynamic_range=(-50, 0))
        block_until_ready(out_b)

        out_c = pipeline_custom(**inputs_single_custom, **params_custom, dynamic_range=(-30, 0))
        block_until_ready(out_c)

        scan.set_transmits(3)
        params_tx = pipeline_custom.prepare_parameters(probe, scan)
        params_tx = dict(params_tx)
        params_tx.pop("dynamic_range", None)
        params_tx.pop("data", None)

        data_11_transmits = data[0][scan.selected_transmits]
        inputs_tx = {pipeline_custom.key: data_11_transmits}

        out_d = pipeline_custom(**inputs_tx, **params_tx)
        block_until_ready(out_d)

        return out_a, out_b, out_c, out_d
    # ----------------------------------------

    # ----------------------------------------
    # 6 | Warm-up
    outs = None
    for _ in range(args.warmup):
        outs = run_task()
    # ----------------------------------------

    # ----------------------------------------
    # 7 | Runs
    import jax.profiler

    run_times = []
    trace_cm = None
    if args.trace_dir:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        trace_cm = jax.profiler.trace(args.trace_dir, create_perfetto_link=False)
        trace_cm.__enter__()

    try:
        for i in range(args.iters):
            with jax.profiler.StepTraceAnnotation("working_task", step_num=i):
                ts = time.perf_counter()
                outs = run_task()
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

    results["tasks_per_call"] = 1
    results["task_rate_mean"] = 1.0 / results["run_time_mean_s"]

    if outs is not None:
        out_a, out_b, out_c, out_d = outs

        def _shape(o, key):
            if isinstance(o, dict) and key in o:
                try:
                    return list(o[key].shape)
                except Exception:
                    return "unknown"
            return None

        results["outputs"] = {
            "default_image_shape": _shape(out_a, pipeline_default.output_key),
            "custom_image_shape_dr50": _shape(out_b, pipeline_custom.output_key),
            "custom_image_shape_dr30": _shape(out_c, pipeline_custom.output_key),
            "custom_image_shape_tx": _shape(out_d, pipeline_custom.output_key),
        }
    # ----------------------------------------

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Benchmark summary ===")
    print(f"Load time:      {results['load_time_s']:.2f} s")
    print(f"Prepare (def):  {results['default_parameters_time_s']:.2f} s")
    print(f"Prepare (cust): {results['custom_parameters_time_s']:.2f} s")
    print(f"Run mean:       {results['run_time_mean_s']:.2f} s  (min {results['run_time_min_s']:.2f} s)")
    print(f"Task rate mean: {results['task_rate_mean']:.2f} (task = full working.py flow)")
    if args.trace_dir:
        print(f"Trace dir:      {args.trace_dir}")
    print(f"Saved JSON:     {args.out}")


if __name__ == "__main__":
    main()
