from __future__ import annotations

from dataclasses import dataclass   # Python 3.7以降で登場した、データクラスを便利に使用するためのモジュール

import torch    # 機械学習のためのライブラリ (PyTorch本体)
from torch import nn    # PyTorchのニューラルネットワークモジュールで、モデルの構築や損失関数などを提供する


# モデルの設定を保持するためのデータクラス
@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 1
    hidden_dims: tuple[int, ...] = (32, 32)
    output_dim: int = 1
    activation: str = "relu"
    weight_init: str = "xavier_uniform"
    hidden_bias_range: float = 0.1


# 活性化関数を構築する関数
def build_activation(name: str) -> nn.Module:
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
    }
    if name not in activations:
        supported = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation: {name}. Choose from: {supported}")
    return activations[name]


class MLPRegressor(nn.Module):
    """ シンプルな全結合ニューラルネットワーク回帰モデル """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()   # 設定が提供されない場合はデフォルトのModelConfigを使用
        # ネットワークの層の次元を定義（入力層、任意の数の隠れ層、出力層）
        dims = [self.config.input_dim, *self.config.hidden_dims, self.config.output_dim]

        layers: list[nn.Module] = []    # ネットワークの層を格納するリスト
        # 隠れ層を構築し、活性化関数を追加
        for in_features, out_features in zip(dims[:-2], dims[1:-1], strict=True):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(build_activation(self.config.activation))
        # 出力層を追加（最後の活性化関数はなし）
        layers.append(nn.Linear(dims[-2], dims[-1]))

        # ネットワークをSequentialでまとめ、扱いやすくする
        self.network = nn.Sequential(*layers)

        # ネットワークの重みとバイアスを初期化
        self.reset_parameters()

    # ネットワークの重みとバイアスを初期化する関数
    def reset_parameters(self) -> None:
        linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        for layer_index, layer in enumerate(linear_layers):
            is_output_layer = layer_index == len(linear_layers) - 1
            self._init_weight(layer.weight)

            if layer.bias is None:  # バイアスがない場合はスキップ
                continue
            if is_output_layer: # 出力層のバイアスはゼロで初期化
                nn.init.zeros_(layer.bias)
            else:   # 隠れ層のバイアスは指定された範囲で一様分布から初期化
                nn.init.uniform_(
                    layer.bias,
                    -self.config.hidden_bias_range,
                    self.config.hidden_bias_range,
                )

    # 重みを初期化する関数
    def _init_weight(self, weight: torch.Tensor) -> None:
        if self.config.weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(weight, gain=self._activation_gain())
        elif self.config.weight_init == "kaiming_uniform":
            nonlinearity, param = self._kaiming_options()
            nn.init.kaiming_uniform_(weight, a=param, nonlinearity=nonlinearity)  # type: ignore
        elif self.config.weight_init == "kaiming_normal":
            nonlinearity, param = self._kaiming_options()
            nn.init.kaiming_normal_(weight, a=param, nonlinearity=nonlinearity)  # type: ignore
        else:
            supported = "xavier_uniform, kaiming_uniform, kaiming_normal"
            raise ValueError(f"Unsupported weight_init: {self.config.weight_init}. Choose from: {supported}")

    # 活性化関数に応じたゲインを計算する関数
    def _activation_gain(self) -> float:
        if self.config.activation == "leaky_relu":
            return nn.init.calculate_gain("leaky_relu", 0.01)
        if self.config.activation in {"relu", "tanh"}:
            return nn.init.calculate_gain(self.config.activation)  # type: ignore
        return 1.0

    # Kaiming初期化のオプションを取得する関数
    def _kaiming_options(self) -> tuple[str, float]:
        if self.config.activation == "relu":
            return "relu", 0.0
        if self.config.activation == "leaky_relu":
            return "leaky_relu", 0.01
        raise ValueError(
            "kaiming_* initialization supports activation='relu' or 'leaky_relu'. "
            f"Current activation is '{self.config.activation}'."
        )

    # ネットワークの順伝播を定義する関数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# 設定辞書からMLPRegressorモデルを構築する関数
def build_model(config_dict: dict | None = None) -> MLPRegressor:
    config_dict = config_dict or {}
    config = ModelConfig(
        input_dim=int(config_dict.get("input_dim", 1)),
        hidden_dims=tuple(int(dim) for dim in config_dict.get("hidden_dims", [32, 32])),
        output_dim=int(config_dict.get("output_dim", 1)),
        activation=str(config_dict.get("activation", "relu")),
        weight_init=str(config_dict.get("weight_init", "xavier_uniform")),
        hidden_bias_range=float(config_dict.get("hidden_bias_range", 0.1)),
    )
    return MLPRegressor(config)


# モデルの動作をテストするための関数
def main() -> None:
    model = MLPRegressor()
    dummy_x = torch.randn(8, 1)
    dummy_y = model(dummy_x)
    assert dummy_y.shape == (8, 1)
    print(f"Forward pass ok: input={tuple(dummy_x.shape)}, output={tuple(dummy_y.shape)}")


# モジュールが直接実行されたときにmain関数を呼び出す
if __name__ == "__main__":
    main()
