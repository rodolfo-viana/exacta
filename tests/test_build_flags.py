"""The native kernels must never be built with floating-point contraction (CMakeLists.txt).

The exactness proof of the distance transform assumes every binary64 product and sum is rounded
once. CMakeLists.txt gives each supported compiler the flags that forbid fused multiply-adds,
stops for any other compiler, and runs a probe (csrc/fp_contract_probe.cpp) with those flags.
"""

import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CMAKELISTS = ROOT / "CMakeLists.txt"
PROBE = ROOT / "csrc" / "fp_contract_probe.cpp"

needs_sources = pytest.mark.skipif(not CMAKELISTS.exists(), reason="needs the source tree")


@needs_sources
def test_cmake_gives_each_supported_compiler_its_flags_and_stops_for_the_others():
    text = CMAKELISTS.read_text()
    # one set of flags per family: GCC and Clang, Intel's icx, MSVC, and the clang-cl frontends
    for flags in ("-ffp-contract=off", "-fp-model=precise -ffp-contract=off", "/fp:precise",
                  "/fp:precise /clang:-ffp-contract=off"):
        assert re.search(rf"set\(EXACTA_FP_FLAGS {re.escape(flags)}\)", text), flags
    chain = text[text.index('if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")'):text.index("# A flag that the compiler")]
    # the last branch is an error: no compiler falls through without flags (the previous test was
    # an unanchored match on "GNU|Clang" that skipped every other compiler silently)
    assert re.search(r"\nelse\(\)\n\s+message\(FATAL_ERROR", chain)
    assert "MATCHES \"GNU|Clang\"" not in text
    assert "target_compile_options(exacta_kernels PRIVATE ${EXACTA_FP_FLAGS})" in text
    assert "fp_contract_probe.cpp" in text and PROBE.exists()


@needs_sources
def test_msvc_before_visual_studio_2022_is_refused():
    # Microsoft documents that /fp:precise may contract before Visual Studio 2022 (cl 19.30)
    text = CMAKELISTS.read_text()
    assert "VERSION_LESS 19.30" in text


def _configure(tmp_path: Path, *flags: str) -> subprocess.CompletedProcess:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("needs cmake")
    nanobind = subprocess.run([sys.executable, "-m", "nanobind", "--cmake_dir"], capture_output=True, text=True)
    if nanobind.returncode:
        pytest.skip("needs nanobind")
    generator = ["-G", "Ninja"] if shutil.which("ninja") else []
    return subprocess.run(
        [cmake, "-S", str(ROOT), "-B", str(tmp_path), *generator, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
         "-DCMAKE_BUILD_TYPE=Release", f"-DPython_EXECUTABLE={sys.executable}",
         f"-Dnanobind_DIR={nanobind.stdout.strip()}", "-DSKBUILD_SABI_COMPONENT=Development.SABIModule", *flags],
        capture_output=True, text=True, timeout=300,
    )


def _kernel_commands(build: Path) -> list[str]:
    path = build / "compile_commands.json"
    if not path.exists():
        pytest.skip("this CMake generator writes no compile_commands.json")
    commands = [e.get("command", " ".join(e.get("arguments", []))) for e in json.loads(path.read_text())]
    kernels = [c for c in commands if re.search(r"csrc[/\\]cpu[/\\][a-z_]+\.cpp", c)]
    assert len(kernels) >= 10
    return kernels


@needs_sources
def test_every_kernel_is_compiled_with_the_contraction_flags(tmp_path):
    done = _configure(tmp_path)
    if done.returncode and "floating-point contraction" not in done.stdout + done.stderr:
        pytest.skip(f"cmake cannot configure in this environment: {(done.stderr or done.stdout)[-300:]}")
    assert done.returncode == 0, done.stdout + done.stderr
    for command in _kernel_commands(tmp_path):
        assert "-ffp-contract=off" in command or "/fp:precise" in command, command


@needs_sources
@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64") or sys.platform == "win32",
                    reason="-march=native with fused multiply-add is the x86-64 case tested here")
def test_the_flags_come_after_the_users_and_override_them(tmp_path):
    done = _configure(tmp_path, "-DCMAKE_CXX_FLAGS=-march=native -ffp-contract=fast")
    if done.returncode and "floating-point contraction" not in done.stdout + done.stderr:
        pytest.skip("cmake cannot configure in this environment")
    assert done.returncode == 0, done.stdout + done.stderr
    for command in _kernel_commands(tmp_path):
        if "-ffp-contract=fast" in command:  # GCC and Clang spell it this way
            assert command.rindex("-ffp-contract=off") > command.rindex("-ffp-contract=fast"), command
