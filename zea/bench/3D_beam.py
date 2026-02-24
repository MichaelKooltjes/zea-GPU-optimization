import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["ZEA_DISABLE_CACHE"] = "1"

def main():
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
    set_mpl_style()

    downscale_rate = 2

    #put this here for easier access
    chunks = 1024 # Increase the number of chunks if you run out of memory
    path = "hf://zeahub/phantoms/2025_12_16_cirs_focused_3d.hdf5"

    rf_data, scan, probe = load_file(
        path=path, 
        indices=[0], 
        data_type="raw_data",
    )

    # index the first frame
    print("RF data shape:", rf_data.shape)

    
    scan.n_ch = 2 # IQ data, should be stored in file but isn't currently
    scan.zlims = (0, 25e-3) # reduce z-limits a bit for better visualization

    #hasattr because it gave me errors for a nonexistent scan.grid_size_y
    if hasattr(scan, "grid_size_x"):
        scan.grid_size_x //= downscale_rate
    if hasattr(scan, "grid_size_y"):
        scan.grid_size_y //= downscale_rate
    if hasattr(scan, "grid_size_z"):
        scan.grid_size_z //= downscale_rate
    print(f"3D grid shape = {scan.grid.shape}")

    pipeline = Pipeline(
        [
            Demodulate(),
            Map(
                [TOFCorrection(), DelayAndSum()],
                argnames="flatgrid",
                chunks=chunks,
            ),
            ReshapeGrid(),
            EnvelopeDetect(),
            Normalize(),
            LogCompress(),
        ],
        with_batch_dim=True,
    )

    parameters = pipeline.prepare_parameters(probe, scan)

    #outputting the data instead of the gif
    out = pipeline(data=rf_data, **parameters)
    print("Output keys:", list(out.keys()))
    if "data" in out:
        try:
            print("Output data shape:", out["data"].shape)
        except Exception:
            pass

if __name__ == "__main__":
    main()
