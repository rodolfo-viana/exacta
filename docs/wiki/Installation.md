# Installation

The released alpha requires Python 3.10 or newer, PyTorch 2.4 or newer and NumPy 1.22 or newer.
Install PyTorch for your hardware using its [installation guide](https://pytorch.org/get-started/locally/),
then install the pinned Exacta release:

```bash
python -m pip install exacta==0.0.1a3
python -c "import exacta; print(exacta.__version__)"
```

Version 0.0.1a3 includes Linux x86-64 wheels for glibc 2.28 or newer. Python 3.10 and 3.11
have dedicated wheels; the Python 3.12 stable-ABI wheel is also tested on Python 3.13 and 3.14.

Other platforms install from source, which builds the CPU extension and requires a C++17
compiler and Python development headers; OpenMP is optional. On Debian/Ubuntu, install
`build-essential` and development headers matching your Python interpreter.
The release workflow builds and tests these wheels; branch CI tests an installation from source.

## CPU and GPU

CPU kernels are precompiled in wheels and compile during source installation. CUDA loss and
metric kernels use Triton at runtime; their first invocation can include compilation. The CPU extension does not link against a
specific PyTorch binary. Local validation covers Linux x86-64 CPU and NVIDIA CUDA execution;
other hardware/platform combinations should be verified before use.

## Optional references

SciPy is needed for Hungarian lesion matching. MONAI, MedPy, VTK, SimpleITK and other reference
libraries are used by optional validation tests; they are not needed for ordinary Dice or
HD95 evaluation. The [development guide](https://github.com/rodolfo-viana/exacta/wiki/Development)
describes test environments.

## Development checkout

Use `main` to evaluate merged changes beyond the published release. Create a feature branch
from it when contributing. Shared metric
evaluation, loss combinations, deep supervision, global distributed losses, ignored labels
and voxel masks are included in 0.0.1a3; they do not require a development checkout.

```bash
git clone --branch main https://github.com/rodolfo-viana/exacta.git
cd exacta
python -m pip install -e ".[test]"
```

Record the commit SHA and environment alongside experiment results. Reinstall after changing
C++ sources so the compiled extension matches the checkout.
