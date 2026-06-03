# PyTorch Machine Learning Chapter 03

PyTorchでノイズ入りの2次関数を学習し、学習中の推論結果をアニメーションとして保存するサンプルです。

## 環境

- Python 3.14.3以上
- uv
- PyTorch CPU版

依存関係は `pyproject.toml` と `uv.lock` で管理しています。

```powershell
uv sync
```

## 実行方法

通常の学習を実行します。

```powershell
uv run python train.py
```

実行すると、以下が出力されます。

- `checkpoints/quadratic_regressor.pt`
- `outputs/quadratic_training.gif`

## 主なファイル

- `train.py`: 学習、チェックポイント保存、アニメーション書き出し
- `configs/train.yaml`: 通常学習用の設定
- `configs/train_overfitting.yaml`: 過学習を観察しやすい設定
- `models/model.py`: 全結合ニューラルネットワーク
- `data/functions.py`: 学習対象の関数とデータ生成

## 設定

ハイパーパラメータや出力先は `configs/train.yaml` で変更できます。

学習対象の関数を変更したい場合は、`data/functions.py` の `target_function()` を編集します。

## 単体確認

モデルのforwardパスを確認します。

```powershell
uv run python -m models.model
```

関数の形状をグラフで確認します。

```powershell
uv run python -m data.functions
```
