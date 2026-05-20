"""L2-aware tiling for DAS beamforming."""

import ctypes
import math
import subprocess

# Fallback table in case the CUDA driver query fails.
# Values are in bytes. Mostly here for machines where libcuda isn't
# directly accessible (e.g. some Docker setups).
_GPU_L2_BYTES = {
    # 30-series
    "RTX 3060 Ti": 3 * 1024 * 1024,
    "RTX 3060":    3 * 1024 * 1024,
    "RTX 3070 Ti": 4 * 1024 * 1024,
    "RTX 3070":    4 * 1024 * 1024,
    "RTX 3080 Ti": 5 * 1024 * 1024,
    "RTX 3080":    5 * 1024 * 1024,
    "RTX 3090 Ti": 6 * 1024 * 1024,
    "RTX 3090":    6 * 1024 * 1024,
    # 40-series
    "RTX 4060 Ti": 32 * 1024 * 1024,
    "RTX 4060":    24 * 1024 * 1024,
    "RTX 4070 Ti": 48 * 1024 * 1024,
    "RTX 4070":    36 * 1024 * 1024,
    "RTX 4080":    64 * 1024 * 1024,
    "RTX 4090":    72 * 1024 * 1024,
    # data-centre
    "A100": 40 * 1024 * 1024,
    "A10":  40 * 1024 * 1024,
    "A30":  24 * 1024 * 1024,
    "V100":  6 * 1024 * 1024,
    "T4":    4 * 1024 * 1024,
}

_DEFAULT_L2_BYTES = 4 * 1024 * 1024  # safe fallback if everything else fails


def query_gpu_l2_cache_bytes(device_idx: int = 0) -> int:
    """Return the L2 cache size in bytes for the given GPU.

    Tries the CUDA driver API first (most reliable), then falls back to
    matching the GPU name from nvidia-smi against the lookup table above,
    and finally just returns 4 MB if nothing works.
    """
    # Ask CUDA directly — attribute 38 is CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
    try:
        lib = ctypes.cdll.LoadLibrary("libcuda.so.1")
        lib.cuInit(0)
        val = ctypes.c_int(0)
        ret = lib.cuDeviceGetAttribute(ctypes.byref(val), 38, device_idx)
        if ret == 0 and val.value > 0:
            return val.value
    except Exception:
        pass

    # CUDA query failed, try matching the GPU name instead
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            names = proc.stdout.strip().split("\n")
            if device_idx < len(names):
                name = names[device_idx].strip()
                for key, size in _GPU_L2_BYTES.items():
                    if key in name:
                        return size
    except Exception:
        pass

    return _DEFAULT_L2_BYTES


def compute_l2_num_patches(
    n_pix: int,
    n_tx: int,
    n_el: int,
    n_ch: int,
    dtype_bytes: int = 4,
    l2_cache_bytes: int | None = None,
    utilization: float = 0.5,
    fused: bool = True,
) -> int:
    """Work out how many patches are needed to keep each tile inside L2.

    The idea: figure out how many bytes of intermediate data each output
    pixel costs, then see how many pixels fit in (utilization * L2), and
    split the grid into enough patches so no tile exceeds that limit.

    For the fused kernel the per-pixel cost is just n_el * n_ch * 4 bytes
    because the element-sum happens immediately inside the per-tx body,
    only one transmit's worth of data is live at a time. For the unfused
    path the full (n_tx, n_el, n_ch) slice is live, so n_tx is included.

    utilization=0.5 leaves room for the delay tables and output buffer.
    """
    if l2_cache_bytes is None:
        l2_cache_bytes = query_gpu_l2_cache_bytes()

    usable = int(l2_cache_bytes * utilization)

    tx_factor = 1 if fused else n_tx
    bytes_per_pixel = tx_factor * n_el * n_ch * dtype_bytes

    n_pix_per_tile = max(1, usable // bytes_per_pixel)
    return max(1, math.ceil(n_pix / n_pix_per_tile))
