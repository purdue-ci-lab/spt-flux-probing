<p align="center">
  <h1 align="center">Spatiotemporal Flux Probing for Single-Photon Videography</h1>
  <p align="center">
    <a href="https://jerukan.github.io/">Jerry Yan</a>
    ·
    <a href="https://www.linkedin.com/in/matteo-forlivesi-92a007252/">Matteo Forlivesi</a>
    ·
    <a href="https://www.linkedin.com/in/bowen-tan-669775183/">Bowen Tan</a>
    ·
    <a href="https://andrewxie.com/">Andrew Xie</a>
    <br>
    <a href="https://sidsoma.com/">Siddharth Somasundaram</a>
    ·
    <a href="https://sotirisnousias.com/">Sotiris Nousias</a>
  </p>
  <h3 align="center">
    <p>ECCV 2026</p>
    <a href="https://jerukan.github.io/spt-flux-probing/" align="center">Project website</a>
    <!-- ·
    <a href="" align="center">Publication</a>
    ·
    <a href="" align="center">Supplementary</a> -->
    
  </h3>
</p>

Official code repository for Spatiotemporal Flux Probing for Single-Photon Videography (ECCV 2026).

## Setup

### Environment

We use [uv](https://docs.astral.sh/uv/) as a package manager to set up the environment.

```shell
uv sync
source .venv/bin/activate
```

To set up CuPy for CUDA acceleration on processing quanta data, check your Nvidia version:

```shell
nvidia-smi
```

Depending on the version, install based on the following:

```shell
# For CUDA 12.x
uv pip install cupy-cuda12x

# For CUDA 13.x
uv pip install cupy-cuda13x
```

If CUDA Toolkit is not installed on the system, append either of the above with `[ctk]`, e.g.

```shell
uv pip install "cupy-cuda12x[ctk]"
```

### Project paths config

Set up paths for data and output in the config folder, so you don't have to keep repeating absolute paths when running things.

Copy [`config/default_paths.yaml`](config/default_paths.yaml) to a new file `config/paths.yaml`. By default, `DATA_DIR` and `OUTPUT_DIR` will be `data` and `output` within the project root. However, if things need to be stored somewhere else, you can specify:

```yaml
DATA_DIR: /path/to/your/data
OUTPUT_DIR: /path/to/write/output
LOG_DIR: /path/to/write/logs
```

Data will automatically be read relative to `DATA_DIR`, and output will always be written relative to `OUTPUT_DIR`.

If you instead prefer to have scripts take in only absolute paths, you can set the paths blank:

```yaml
DATA_DIR: ''
OUTPUT_DIR: ''
```

### Sample data download

Download the data from Dropbox [here](https://www.dropbox.com/scl/fo/jwm9bcgbc87e8xib4koyd/AIk2M1QYeYToMTx_bqond5A?rlkey=a590dszivimivn9sb0z6zh08s&st=ab7e8w2f&dl=0).

Put them in your `DATA_DIR`.

**Note**: Any quanta data stored as a `.zarr` format has already been hot-pixel corrected. There may still be some weaker hot pixels still visible, but they should not be able to ruin dynamic range.

## Running the code

### Quanta videography

Quanta data are data that is frame-based (i.e. one event takes one pixel, and frames are sampled at an even rate). This consists of both quanta SPAD sensors and neuromorphic spike cameras.

**Supported data formats**: The primary format we use for data are `.zarr` files, since they are stored compressed and can be read lazily. Other supported file formats are the `.bin` files or directories of `.bin` files as outputted by default by Pi Imaging SPAD 512 or SPAD Alpha cameras. Alternatively, `.mat` formats as used in [Quanta Burst Photography](https://github.com/sizhuom/quanta-burst-photography) are also supported.

**Data configuration**: Set up the data dimensions and framerate within `config/data_types`, where you can reference existing configurations for the SPAD 512 camera ([`config/data_types/spad512_100k.yaml`](config/data_types/spad512_100k.yaml)) and the spike camera ([`config/data_types/spike_40k.yaml`](config/data_types/spike_40k.yaml)).

**Run configuration**: Configure spatiotemporal flux probing and reconstruction parameters within `config/runs`. An example run configuration for SPAD 512 data is [`config/runs/example_quanta.yaml`](config/runs/example_quanta.yaml).

**Note**: Ensure you define `recon_slice` in the run configuration, either in the run config directly or in [`config/recon_slices.yaml`](config/recon_slices.yaml) if you're reconstructing at a high framerate to limit the number of frames you reconstruct. Otherwise you may output a 100000 x 512 x 512 reconstrucion that can take over 100 GB of space.

The script to run quanta videography is [`run_uwb3d_quanta.py`](run_uwb3d_quanta.py).

```
usage: run_uwb3d_quanta.py [-h] --run RUN [--recon-slices RECON_SLICES] [--device DEVICE] [--cuda] [--no-vram-frames] [--no-vram-output] [--zip] [--save-flux-vid] [--save-gamma-pngs]

Windowed spatiotemporal flux probing with via STFT for quanta videos.

options:
  -h, --help            show this help message and exit
  --run RUN             Path to a run-parameter YAML (config/runs/*.yaml).
  --recon-slices RECON_SLICES
                        Path to the per-dataset recon-slices YAML (default: config/recon_slices.yaml).
  --device DEVICE       CUDA device index.
  --cuda                Use CUDA if a GPU is available (default: CPU).
  --no-vram-frames      Keep input frames in host memory instead of VRAM.
  --no-vram-output      Keep the output in host memory instead of VRAM.
  --zip                 Zip each run's output dir into the transfer dir when done.
  --save-flux-vid       Also save a video of the flux (default: only save gamma-corrected video).
  --save-gamma-pngs     Also save a folder of gamma-corrected PNGs (default: only save video).
```

An example usage is the following:

```shell
uv run run_uwb3d_quanta.py --run example_quanta.yaml --device 0 --cuda --no-vram-frames --no-vram-output
```

If you have enough VRAM on your GPU, exclude the arguments `--no-vram-frames` or `--no-vram-output`, and the code will run significantly faster.

**Note**: Smaller datasets like the spike camera datasets or slices of the quanta data are fully runnable on CPU! A GPU is not required unless you are doing large window sizes with high overlap on full volumes of quanta data.

By default, this will only output `gamma.mp4` (reconstruction with a gamma correction of 2.2) and `reconinfo.zarr` (full float array of output flux).

### Asynchonrous SPAD and ultra-wideband videography

Since timestamping SPAD arrays record picosecond-resolution timestamps and not frames, we treat their data as a 3D point cloud of continuous-position photons instead.

An example of running spatiotemporal flux probing + reconstruction on continuous point data is in [`example_asyncspad.py`](example_asyncspad.py) (it's full of Claude comment slop, but you should be able to use it as reference), which you can run with

```shell
uv run example_asyncspad.py
```

The real data we run this on for ultra-wideband videography is on data from [Passive Ultra-Wideband](https://github.com/t-cig/UWB1D). This data should be available on our Dropbox data download under `AC_fan`.

The scripts to run reconstruction on this data is in [`uwbfan_forward.py`](uwbfan_forward.py) and [`uwbfan_recon.py`](uwbfan_recon.py). These scripts use harmonic probing for the known laser frequency instead of an exhaustive search of all frequencies due to computational limitations. See the corresponding files for what to modify in them before running.

```shell
# generate the Fourier probing measurements
uv run uwbfan_forward.py
# reconstruct at various framerates from probing measurements
uv run uwbfan_recon.py
```

### Velocity-selective videography

TODO Teo write this pls.

### Event camera videography

The code for event camera videography is in [`scratch_event.ipynb`](scratch_event.ipynb). The code is a complete mess and requires a lot of manual changes per dataset. I do not plan to clean up this code, but if you really need a more usable version of this code, you can contact me. Alternatively, Claude can probably(?) make some sense of it.
