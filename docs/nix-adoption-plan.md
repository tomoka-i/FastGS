# Nix Adoption Plan for FastGS

FastGS currently relies on a Conda environment with Python 3.7.13, PyTorch
1.12.1, CUDA 11.6 runtime packages, and three local CUDA/PyTorch extensions.
The Nix setup should be introduced incrementally so the existing Conda workflow
continues to work while Nix support is validated.

## Goals

- Provide a reproducible developer shell for Rust extension work.
- Preserve compatibility with the existing `environment.yml` workflow.
- Make CUDA/PyTorch extension builds explicit and documented.
- Avoid changing training behavior until the Nix shell is proven on a CUDA host.

## Constraints

- PyTorch is pinned to `1.12.1`.
- The project expects Python `3.7.13`.
- CUDA extension builds depend on a compatible C++ compiler and CUDA toolkit.
- Local pip dependencies point at repository submodules:
  - `submodules/diff-gaussian-rasterization_fastgs`
  - `submodules/simple-knn`
  - `submodules/fused-ssim`
- macOS can validate Rust/Python scaffolding, but full CUDA training validation
  requires a Linux CUDA machine.

## Checklist

- [ ] Confirm target Nix platform for training validation.
  - Recommended: Linux with NVIDIA driver and CUDA-capable GPU.
- [ ] Decide whether Nix should manage PyTorch directly or provide a shell around
      the existing Conda environment.
  - Recommended first step: Nix shell for tooling only.
- [ ] Add `flake.nix` with a minimal dev shell.
  - Include Rust toolchain.
  - Include `pkg-config`, C/C++ compiler, Python tooling, and `maturin`.
  - Keep CUDA/PyTorch as explicit follow-up work.
- [ ] Add `.envrc` for `direnv` users.
  - Use `use flake` after the flake is stable.
- [ ] Document bootstrap commands in `README.md` or a dedicated Nix doc.
  - `nix develop`
  - `cargo check`
  - `maturin develop --release`
- [ ] Validate Rust extension workflow inside Nix.
  - `cd csrc_rust`
  - `cargo fmt --check`
  - `cargo check`
  - `maturin develop --release`
- [ ] Map the Conda dependencies to Nix equivalents.
  - Python 3.7
  - PyTorch 1.12.1
  - torchvision 0.13.1
  - torchaudio 0.12.1
  - cudatoolkit 11.6
  - tqdm
  - websockets
  - plyfile
- [ ] Investigate PyTorch 1.12.1 availability in the selected nixpkgs pin.
  - If unavailable, evaluate overlay, binary wheel install, or Conda-in-Nix.
- [ ] Validate local CUDA extension builds.
  - `submodules/diff-gaussian-rasterization_fastgs`
  - `submodules/simple-knn`
  - `submodules/fused-ssim`
- [ ] Run import smoke tests.
  - `python -c "import torch"`
  - `python -c "import fastgs_rust"`
  - `python -c "from gaussian_renderer import render_fastgs"`
- [ ] Run a minimal training smoke test on a small dataset.
  - Confirm CUDA visibility.
  - Confirm custom rasterizer import.
  - Confirm no compiler/runtime library mismatch.
- [ ] Decide whether to replace or keep `environment.yml`.
  - Recommended: keep both until Nix training is fully validated.

## Recommended First Implementation

Start with a tooling-only `flake.nix`:

- Rust compiler and Cargo for `csrc_rust`.
- Python with `pip`.
- `maturin` for building the PyO3 module.
- Common native build tools.

This gets the Rust work reproducible without forcing the older PyTorch/CUDA
stack through Nix immediately. After that works, add a second Linux CUDA-focused
dev shell for training.

## Validation Notes

The first meaningful success criteria are:

1. `cargo check` passes in `csrc_rust`.
2. `maturin develop --release` installs `fastgs_rust`.
3. A Python smoke test can call `fastgs_rust.extension_info()`.

Full training validation should be treated separately because it depends on
CUDA, NVIDIA driver compatibility, and the existing PyTorch extension submodules.
