# Nix Development Shell

This repository now includes a tooling-only Nix shell. It is intentionally not a
full replacement for `environment.yml` yet.

## Why Tooling-Only First

FastGS depends on Python 3.7.13, PyTorch 1.12.1, CUDA 11.6, and several local
CUDA/PyTorch extensions. Reproducing that whole training stack in Nix should be
validated on a Linux CUDA machine before replacing the existing Conda workflow.

The first Nix shell focuses on lower-risk developer tooling:

- Rust compiler and Cargo
- `rustfmt` and `clippy`
- Python with `pip`, `setuptools`, `wheel`
- `maturin` for PyO3 extension builds
- native build helpers such as `pkg-config`

## Usage

```bash
nix develop
```

If you use `direnv`:

```bash
direnv allow
```

## Rust Extension Workflow

After the Rust extension branch is merged or checked out:

```bash
nix develop
cd csrc_rust
cargo fmt --check
cargo check
maturin develop --release
```

## Training Environment

Continue using the Conda environment for training for now:

```bash
conda env create --file environment.yml
conda activate fastgs
```

Moving training to Nix is a separate follow-up task because it requires checking
PyTorch 1.12.1, CUDA 11.6, NVIDIA driver compatibility, and local extension ABI
compatibility together.
