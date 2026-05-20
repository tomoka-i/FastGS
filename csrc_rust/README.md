# fastgs_rust

Rust extension scaffold for FastGS score computation experiments.

This module is intentionally small and safe to import from the existing Python
training code. It does not link against libtorch yet; instead, it accepts
Python `torch.Tensor` objects through PyO3 and calls tensor methods on them.
That keeps the first Rust boundary easy to build in the current FastGS
environment while leaving `src/lib.rs` as the place to replace Python tensor
method calls with `tch`/libtorch code later.

## Build

From the repository root:

```bash
pip install maturin
cd csrc_rust
maturin develop --release
```

On Colab, install Rust first:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
```

## Smoke Test

```bash
python - <<'PY'
import torch
import fastgs_rust

counts = [torch.tensor([1, 2, 3]), torch.tensor([2, 2, 6])]
losses = [torch.tensor(0.5), torch.tensor(1.0)]
importance, pruning = fastgs_rust.compute_scores_from_accumulators(counts, losses, True)
print(importance)
print(pruning)
PY
```

## Next Step

`utils.fast_utils.compute_gaussian_score_fastgs` still performs rendering in
Python. The useful next Rust target is the accumulation block after each
`render_fastgs(..., get_flag=True, metric_map=metric_map)` call:

```python
full_metric_score += photometric_loss * accum_loss_counts
full_metric_counts += accum_loss_counts
```

Once this PyO3 boundary is verified in the training environment, replace the
helper functions in `src/lib.rs` with direct tensor operations through a
libtorch-compatible Rust crate.
