from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from data.functions import build_function_config, target_function
from models.model import build_model
from train import build_paths, load_config


# 学習の設定ファイルのパスを定義する。ここでは、configsディレクトリにあるtrain.yamlというファイルを指定している。
CONFIG_PATH = Path("configs/train.yaml")


# 学習済みモデルをロードする関数を定義する。この関数は、設定ファイルからモデルの構築方法を読み取り、指定されたチェックポイントからモデルの重みをロードして、指定されたデバイスに配置する。
def load_trained_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train.py first to create it."
        )

    model = build_model(config["model"])    # モデルの構築方法を設定ファイルから読み取ってモデルを構築する
    checkpoint = torch.load(checkpoint_path, map_location=device)   # 指定されたチェックポイントからモデルの重みをロードする
    model.load_state_dict(checkpoint["model_state_dict"])   # モデルの重みをロードする
    model.to(device)   # モデルを指定されたデバイスに移動する
    model.eval()   # モデルを評価モードに設定する
    return model


# 学習済みモデルを使用して、指定された関数の予測値を計算し、結果を表示する。
def print_predictions(config_path: Path) -> None:
    config = load_config(config_path)
    device = torch.device(config.get("device", "cpu"))
    function_config = build_function_config(config["data"])
    paths = build_paths(config)

    model = load_trained_model(config, paths.checkpoint_path, device)   # 学習済みモデルをロードする

    x = torch.linspace(function_config.x_min, function_config.x_max, steps=20).unsqueeze(1) # 設定の範囲内で20個の等間隔の値を生成し、これをモデルの入力として使用する
    y_true = target_function(x, function_config)    # 生成した入力に対して、ターゲット関数を使用して正解の出力を計算する

    with torch.no_grad():   # 勾配計算を無効にする
        y_pred = model(x.to(device)).cpu()  # モデルの予測値を計算し、CPUに移動する

    # 生成した入力、正解の出力、モデルの予測値を表形式で表示する
    print(f"{'x':>10} {'target':>14} {'model(x)':>14}")
    print("-" * 40)
    for x_value, true_value, pred_value in zip(
        x.squeeze(1).tolist(),
        y_true.squeeze(1).tolist(),
        y_pred.squeeze(1).tolist(),
        strict=True,
    ):
        print(f"{x_value:10.4f} {true_value:14.6f} {pred_value:14.6f}")


# メイン関数。ここでは、print_predictions関数を呼び出して、予測値を表示する。
def main() -> None:
    print_predictions(CONFIG_PATH)


# スクリプトが直接実行されたときにmain関数を呼び出す
if __name__ == "__main__":
    main()
