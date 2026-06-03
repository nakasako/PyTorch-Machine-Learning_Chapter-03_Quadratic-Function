from __future__ import annotations

from dataclasses import dataclass   # Python 3.7 以降で登場した、データクラスを便利に使用するためのモジュール

import matplotlib.pyplot as plt     # グラフ描画のためのライブラリ
import torch                 # 機械学習のためのライブラリ (PyTorch本体)


# 関数の設定を保持するためのデータクラス
@dataclass(frozen=True)     
class FunctionConfig:   
    num_samples: int = 256  # サンプル数
    x_min: float = -3.0     # xの最小値
    x_max: float = 3.0      # xの最大値
    noise_std: float = 0.35 # ノイズの標準偏差（ノイズの量）
    a: float = 0.7          # 二次関数の係数a
    b: float = -0.4         # 二次関数の係数b
    c: float = 1.2          # 二次関数の係数c


# 学習データを生成する関数
def target_function(x: torch.Tensor, config: FunctionConfig) -> torch.Tensor:
    """ 別の関数を学習させるときは、ここを変更 """
    return config.a * x**2 + config.b * x + config.c    # ごく一般的なな二次関数を定義


# データセットを生成する関数
def generate_dataset(config: FunctionConfig, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()   # 乱数生成器を作成
    if seed is not None:
        generator.manual_seed(seed)  # シードを設定して再現性を確保

    x = torch.linspace(config.x_min, config.x_max, config.num_samples).unsqueeze(1) # xを等間隔で生成
    y_clean = target_function(x, config) # 等間隔のxを関数に通して、ノイズのないyを生成
    noise = torch.randn(y_clean.shape, generator=generator) * config.noise_std # ノイズを生成
    return x, y_clean + noise # xおよびノイズを加えたyを返す


# 設定辞書からFunctionConfig(関数の設定が格納されたオブジェクト)を構築する関数
def build_function_config(config_dict: dict | None = None) -> FunctionConfig:
    config_dict = config_dict or {}
    quadratic = config_dict.get("quadratic", {})
    return FunctionConfig(
        num_samples=int(config_dict.get("num_samples", 256)),
        x_min=float(config_dict.get("x_min", -3.0)),
        x_max=float(config_dict.get("x_max", 3.0)),
        noise_std=float(config_dict.get("noise_std", 0.35)),
        a=float(quadratic.get("a", 0.7)),
        b=float(quadratic.get("b", -0.4)),
        c=float(quadratic.get("c", 1.2)),
    )


# 関数のグラフを描画する関数
def plot_function(config: FunctionConfig) -> None:
    x = torch.linspace(config.x_min, config.x_max, 500).unsqueeze(1)
    y = target_function(x, config)
    train_x, train_y = generate_dataset(config, seed=0)

    plt.figure(figsize=(8, 5))
    plt.plot(x.squeeze().numpy(), y.squeeze().numpy(), label="target function", color="tab:blue")
    plt.scatter(train_x.squeeze().numpy(), train_y.squeeze().numpy(), label="noisy samples", s=12, alpha=0.55)
    plt.title("Target Function Preview")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# メイン関数。ここでは、FunctionConfigのデフォルト設定を使用して関数のグラフを描画する。
def main() -> None:
    plot_function(FunctionConfig())


# スクリプトが直接実行されたときにmain関数を呼び出す
if __name__ == "__main__":
    main()
