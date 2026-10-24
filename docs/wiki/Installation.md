# Installation

The released alpha requires Python 3.10 or newer, PyTorch 2.4 or newer and NumPy 1.22 or newer.
Install PyTorch for your hardware using its [installation guide](https://pytorch.org/get-started/locally/),
then install the pinned Exacta release:

```bash
python -m pip install exacta==0.0.1a2
python -c "import exacta; print(exacta.__version__)"
```

Version 0.0.1a2 is distributed as source. Installation builds the CPU extension and requires
a C++17 compiler and Python development headers; OpenMP is optional. On Debian/Ubuntu,
install `build-essential` and development headers matching your Python interpreter.
Portable prebuilt wheels are not yet published.

The `dev` CI workflow builds portable Linux x86-64 wheel artifacts for glibc 2.28 or newer.
It covers Python 3.10 and 3.11 and a stable-ABI wheel built on 3.12, also tested on 3.13/3.14.
These are development artifacts; PyPI will gain wheels in the next release. Other platforms
continue to require a source build.

## CPU and GPU

CPU kernels compile during installation. CUDA loss and metric kernels use Triton at runtime;
their first invocation can include compilation. The CPU extension does not link against a
specific PyTorch binary. Local validation covers Linux x86-64 CPU and NVIDIA CUDA execution;
other hardware/platform combinations should be verified before use.

## Optional references

SciPy is needed for Hungarian lesion matching. MONAI, MedPy, VTK, SimpleITK and other reference
libraries are used by optional validation tests; they are not needed for ordinary Dice or
HD95 evaluation. The [development guide](https://github.com/rodolfo-viana/exacta/wiki/Development)
describes test environments.

## Development checkout

Unreleased features, including shared metric evaluation, loss combinations, deep supervision,
global distributed losses, ignored labels and voxel masks, live on `dev`.
Install that branch explicitly when evaluating those features:

```bash
git clone --branch dev https://github.com/rodolfo-viana/exacta.git
cd exacta
python -m pip install -e ".[test]"
```

Record the commit SHA and environment alongside experiment results. Reinstall after changing
C++ sources so the compiled extension matches the checkout.
