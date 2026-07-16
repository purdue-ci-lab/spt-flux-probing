# photon-testing-grounds

3D flux probing.

## Setup

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
pip install cupy-cuda12x

# For CUDA 13.x
pip install cupy-cuda13x
```

If CUDA Toolkit is not installed on the system, append either of the above with `[ctk]`, e.g.

```shell
pip install "cupy-cuda12x[ctk]"
```
