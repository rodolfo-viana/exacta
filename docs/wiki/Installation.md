# Installation

The released alpha requires Python 3.10 or newer, PyTorch 2.4 or newer and NumPy 1.22 or newer.
Install PyTorch for your hardware using its [installation guide](https://pytorch.org/get-started/locally/),
then install the pinned Exacta release:

```bash
python -m pip install exacta==0.2.0
python -c "import exacta; print(exacta.__version__)"
```

Version 0.2.0 includes Linux x86-64 wheels for glibc 2.28 or newer. Python 3.10 and 3.11
have dedicated wheels; the Python 3.12 stable-ABI wheel supports Python 3.12 and newer.

Other platforms install from source, which builds the CPU extension and requires a C++17
compiler and Python development headers; OpenMP is optional. On Debian/Ubuntu, install
`build-essential` and development headers matching your Python interpreter.

The kernels reproduce reference expressions bit for bit, so they must be compiled without fused
multiply-adds. The build gives GCC, Clang (Apple clang included), Intel icx and MSVC 2022 or later
the flags that forbid them, checks at configuration time that the compiler does round every product
and sum separately, and stops with an error for any other compiler or when that check fails. GCC,
Clang and Apple clang are tested in CI; the MSVC and icx flags follow the vendors' documentation and
have not been exercised.

## CPU and GPU

CPU kernels are precompiled in wheels and compile during source installation. CUDA loss and
metric kernels use Triton at runtime; their first invocation can include compilation. The CPU extension does not link against a
specific PyTorch binary. Local validation covers Linux x86-64 CPU and NVIDIA CUDA execution;
other hardware/platform combinations should be verified before use.

## Distributed CPU execution

For Gloo distributed runs, use PyTorch 2.8 or newer. PyTorch 2.4.1 can deadlock during
process-group destruction after collecting metric scores: a Gloo worker needs the Python
GIL while the outer destructor holds it. PyTorch's [2.8 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.8.0)
include the destructor fix (#154976). Ordinary non-distributed CPU evaluation still supports
PyTorch 2.4.

## Optional references

SciPy is needed for Hungarian lesion matching. MONAI, MedPy, VTK, SimpleITK and other reference
libraries are used by optional validation tests; they are not needed for ordinary Dice or
HD95 evaluation. The [development guide](https://github.com/rodolfo-viana/exacta/wiki/Development)
describes test environments.

## Development checkout

Use `main` to evaluate merged changes beyond the published release. Create a feature branch
from it when contributing. These pages describe version 0.2.0; a newer checkout may expose
additional APIs or changed behavior.

```bash
git clone --branch main https://github.com/rodolfo-viana/exacta.git
cd exacta
python -m pip install -e ".[test]"
```

Record the commit SHA and environment alongside experiment results. Reinstall after changing
C++ sources so the compiled extension matches the checkout.
