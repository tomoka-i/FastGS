アップロードしていただいたGitHubリポジトリ（`FastGS`）の構成に合わせて、**「C++拡張」と「Rust拡張」をプロジェクトに組み込む手順を1から解説**します。

得意とされているC++でのデータ構造やアルゴリズム、競技プログラミングのアプローチをそのままPyTorchのC++ API（ATenライブラリ）に応用できるため、このオフロード作業はかなりスムーズに進められるはずです。

今回は対象となる `utils/fast_utils.py` 内の `compute_gaussian_score_fastgs` の処理をそれぞれオフロードするための「枠組み（フレームワーク）」を作ります。

---

### Step 1: C++ 拡張モジュールの作成 (PyTorch C++ Extension)

PyTorch公式のC++拡張機能を利用します。リポジトリの直下に `csrc_cpp` というフォルダを作成し、以下の2つのファイルを配置します。

#### 1-1. `csrc_cpp/setup.py` の作成

Colabなどの環境でビルドするための設定ファイルです。

```python
# csrc_cpp/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fastgs_cpp',
    ext_modules=[
        CppExtension(
            name='fastgs_cpp', # Pythonからimportするモジュール名
            sources=['compute_score.cpp'],
            extra_compile_args=['-O3'] # 最適化フラグ
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

```

#### 1-2. `csrc_cpp/compute_score.cpp` の作成

計算ロジックの本体です。Pythonからテンソルを受け取り、C++側で計算を行って結果を返します。

```cpp
// csrc_cpp/compute_score.cpp
#include <torch/extension.h>
#include <vector>

// スコア計算のC++実装
std::tuple<torch::Tensor, torch::Tensor> compute_score_cpp(
    const torch::Tensor& xyz,       // [N, 3] などの入力テンソル
    const torch::Tensor& opacity,   // [N, 1] などの入力テンソル
    bool densify) 
{
    // === ここに fast_utils.py の計算ロジックを移植します ===
    // ATen APIを使用するため、慣れ親しんだC++の感覚でテンソル操作が可能です。
    // 例: auto importance_score = xyz.pow(2).sum(1, true) * opacity;
    
    // 以下はコンパイルを通すためのダミー結果です
    auto importance_score = torch::ones({xyz.size(0), 1}, xyz.options());
    auto pruning_score = torch::zeros({xyz.size(0), 1}, xyz.options());

    return std::make_tuple(importance_score, pruning_score);
}

// Python側にバインド（公開）する設定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_score", &compute_score_cpp, "Compute Gaussian Score (C++)");
}

```

---

### Step 2: Rust 拡張モジュールの作成 (PyO3 + tch-rs)

比較用のRust拡張機能も準備します。リポジトリの直下に `csrc_rust` フォルダを作成します。

#### 2-1. `csrc_rust/Cargo.toml` の作成

Rustのパッケージ設定ファイルです。

```toml
[package]
name = "fastgs_rust"
version = "0.1.0"
edition = "2021"

[lib]
name = "fastgs_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.19.0", features = ["extension-module"] }
tch = "0.13.0" # PyTorchのC++ APIをRustから叩くためのクレート

```

#### 2-2. `csrc_rust/pyproject.toml` の作成

Pythonモジュールとしてビルドするためのファイルです。

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "fastgs_rust"
version = "0.1.0"

```

#### 2-3. `csrc_rust/src/lib.rs` の作成

（`csrc_rust` の中に `src` フォルダを作り、その中に配置します）

```rust
// csrc_rust/src/lib.rs
use pyo3::prelude::*;
// use tch::Tensor; // 実際の計算で利用します

#[pyfunction]
fn compute_score_rust(_py: Python, /* 入力テンソルを受け取る処理 */ densify: bool) -> PyResult<String> {
    
    // === ここにRust (tch-rs) の計算ロジックを実装します ===
    
    Ok("Rust execution completed".to_string())
}

#[pymodule]
fn fastgs_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_score_rust, m)?)?;
    Ok(())
}

```

---

### Step 3: Python側 (`fast_utils.py`) での置き換え

元のPythonコードの処理を、作成した拡張モジュールへ分岐させるように変更します。

```python
# utils/fast_utils.py の冒頭でインポート
import torch

# ビルドしたC++モジュールのインポート
try:
    import fastgs_cpp
    C_EXT_AVAILABLE = True
except ImportError:
    C_EXT_AVAILABLE = False

# import fastgs_rust # Rust版

def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True):
    
    if C_EXT_AVAILABLE:
        # === C++拡張に処理をオフロード ===
        # C++側で定義した引数に合わせて必要なテンソルを渡します
        importance_score, pruning_score = fastgs_cpp.compute_score(
            gaussians.get_xyz, 
            gaussians.get_opacity, 
            DENSIFY
        )
        return importance_score, pruning_score
    else:
        # === 元のPython (PyTorch) の処理 ===
        # ... (元の計算ロジック) ...
        pass

```

---

### Step 4: Google Colab でのビルドと実行手順

GitHubからColab環境にクローンした後、学習を開始する**前**のセルでこれらの拡張機能をコンパイルしてインストールします。

```bash
# 1. リポジトリへ移動
%cd /content/FastGS

# 2. C++拡張のコンパイルとインストール
!cd csrc_cpp && pip install -e .

# 3. Rust拡張のコンパイルとインストール (Maturinを使用)
# ※ColabにはRustコンパイラが入っていないためインストールが必要です
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
import os
os.environ['PATH'] += ":/root/.cargo/bin"
!pip install maturin
!cd csrc_rust && maturin develop --release

# 4. 通常通りの学習スクリプト実行
!python train.py -s <データセットのパス>

```

### 実装を進めるステップ

いきなり複雑な計算をC++に移すのではなく、まずは「入ってきたテンソルの形（Shape）や要素数を `std::cout` で出力して、ダミーのゼロテンソルをそのまま返すだけ」の処理を書き、PythonからC++へ正しくデータが渡っているかを確認する（疎通確認）ところから始めると確実です。

`fast_utils.py` の `compute_gaussian_score_fastgs` 内部にある具体的なPythonの数式やループ処理があれば、それを**C++ (ATen API) のコードへ翻訳**することも可能ですので、枠組みができたら実際の処理の中身も確認していきましょう！

---

## 実装済み: Rust 拡張の初期 scaffold

このリポジトリには、Rust 実装を進めるための最小構成として `csrc_rust/` を追加済みです。

追加ファイル:

- `csrc_rust/Cargo.toml`
- `csrc_rust/pyproject.toml`
- `csrc_rust/src/lib.rs`
- `csrc_rust/README.md`

現時点では `tch`/libtorch へ直接リンクせず、PyO3 経由で Python の `torch.Tensor` オブジェクトを受け取る構成です。FastGS の環境は Python 3.7 + PyTorch 1.12.1 系なので、最初から `tch` を入れると libtorch 互換バージョンで詰まりやすいためです。まずは Rust 拡張として build/import できる境界を作り、その後 `src/lib.rs` の中身を直接 tensor 実装へ置き換えるのが安全です。

提供 API:

- `fastgs_rust.normalize_score(score)`
- `fastgs_rust.average_counts(counts, num_views)`
- `fastgs_rust.compute_scores_from_accumulators(accum_metric_counts, photometric_losses, densify)`
- `fastgs_rust.extension_info()`

ビルド:

```bash
cd /Users/somomma/ghq/github.com/tomoka-i/FastGS/csrc_rust
cargo check
pip install maturin
maturin develop --release
```

Colab では先に Rust を入れます。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
```

疎通確認:

```bash
python - <<'PY'
import torch
import fastgs_rust

counts = [torch.tensor([1, 2, 3]), torch.tensor([2, 2, 6])]
losses = [torch.tensor(0.5), torch.tensor(1.0)]
importance, pruning = fastgs_rust.compute_scores_from_accumulators(counts, losses, True)
print(importance)
print(pruning)
print(fastgs_rust.extension_info())
PY
```

次に Rust 化する対象は、`utils/fast_utils.py` の `compute_gaussian_score_fastgs` 内にある以下の集約処理です。

```python
full_metric_counts += accum_loss_counts
full_metric_score += photometric_loss * accum_loss_counts
pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score))
importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
```

レンダリング自体は `render_fastgs` の CUDA 拡張に依存しているため、Rust 側へ移すよりも、まずこの score accumulation 境界から置き換えるのが現実的です。
