from __future__ import annotations

import copy  # Pythonの組み込みモジュールで、オブジェクトのコピーを作成するために使用される。
import random   # Pythonの組み込みモジュールで、乱数生成やシードの設定などを行うために使用される
from dataclasses import dataclass   # Python 3.7以降で登場した、データクラスを便利に使用するためのモジュール
from pathlib import Path    # ファイルシステムのパスを操作するためのモジュール。Pathクラスを使用して、ファイルやディレクトリのパスを表現し、操作することができる。
from typing import Any  # Pythonの型ヒントで、任意の型を表すために使用される特殊な型

import matplotlib.pyplot as plt    # グラフ描画のためのライブラリ
import torch    # 機械学習のためのライブラリ (PyTorch本体)
import yaml     # YAMLファイルを読み書きするためのライブラリ
import imageio_ffmpeg   # FFmpegをPythonから簡単に使用できるようにするためのラッパー (FFmpegのインストールが必要な可能性あり)
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter  # アニメーションを作成するためのMatplotlibのモジュール
from torch import nn    # PyTorchのニューラルネットワークモジュールで、モデルの構築や損失関数などを提供する

from data.functions import build_function_config, generate_dataset, target_function
from models.model import build_model

# 学習の設定ファイルのパスを定義する。ここでは、configsディレクトリにあるtrain.yamlというファイルを指定している。
CONFIG_PATH = Path("configs/train.yaml")
# CONFIG_PATH = Path("configs/train_overfitting.yaml")  # 過学習を体感できるワクワクドキドキ設定ファイル


# 学習に関連するパスや設定をまとめるためのデータクラス
@dataclass(frozen=True)
class TrainPaths:
    checkpoint_dir: Path    # チェックポイントを保存するディレクトリのパス
    checkpoint_name: str    # チェックポイントのファイル名
    output_dir: Path        # アニメーションなどの出力を保存するディレクトリのパス
    animation_name: str     # アニメーションのファイル名

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name # チェックポイントの完全なパスを返す (checkpoint_dirとcheckpoint_nameを組み合わせる)

    @property
    def animation_path(self) -> Path:
        return self.output_dir / self.animation_name # アニメーションの完全なパスを返す (output_dirとanimation_nameを組み合わせる)


# 学習の設定をまとめるためのデータクラス
@dataclass(frozen=True)
class TrainingConfig:
    epochs: int    # 学習のエポック数
    learning_rate: float    # 学習率
    weight_decay: float    # 重み減衰 (L2正則化)
    log_every: int    # ログを出力する間隔 (エポック数)
    snapshot_every: int    # スナップショットを保存する間隔 (エポック数)


# アニメーションの設定をまとめるためのデータクラス
@dataclass(frozen=True)
class AnimationConfig:
    fps: int    # アニメーションのフレームレート
    dpi: int    # アニメーションの解像度


# 設定ファイルを読み込む関数
def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) # YAMLファイルを読み込んで、Pythonの辞書として返す


# 乱数シードを設定する関数
def set_seed(seed: int) -> None:
    random.seed(seed) # Pythonの組み込みrandomモジュールのシードを設定
    torch.manual_seed(seed) # PyTorchのCPUでの乱数シードを設定
    # PyTorchのGPUでの乱数シードを設定し、再現性を確保するための設定を行う
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # PyTorchのGPUでの乱数シードを設定
        torch.backends.cudnn.deterministic = True # CuDNNの動作を決定的にする
        torch.backends.cudnn.benchmark = False # CuDNNのベンチマークを無効にして、再現性を確保


# 設定辞書からTrainPaths(学習に関連するパスが格納されたオブジェクト)を構築する関数
def build_paths(config: dict[str, Any]) -> TrainPaths:
    paths = config["paths"]
    return TrainPaths(
        checkpoint_dir=Path(paths["checkpoint_dir"]),
        checkpoint_name=str(paths["checkpoint_name"]),
        output_dir=Path(paths["output_dir"]),
        animation_name=str(paths["animation_name"]),
    )


# 設定辞書からTrainingConfig(学習の設定が格納されたオブジェクト)を構築する関数
def build_training_config(config: dict[str, Any]) -> TrainingConfig:
    training = config["training"]
    return TrainingConfig(
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        log_every=int(training["log_every"]),
        snapshot_every=int(training["snapshot_every"]),
    )


# 設定辞書からAnimationConfig(アニメーションの設定が格納されたオブジェクト)を構築する関数
def build_animation_config(config: dict[str, Any]) -> AnimationConfig:
    animation = config["animation"]
    return AnimationConfig(fps=int(animation["fps"]), dpi=int(animation["dpi"]))


# 学習を実行する関数。モデルの学習ループを実装し、損失の履歴とスナップショットを保存する。
def train_model(
    model: nn.Module,   # 学習させるモデル
    x: torch.Tensor,    # 学習データの入力 (x)
    y: torch.Tensor,    # 学習データの出力 (y)
    config: TrainingConfig, # 学習の設定
    device: torch.device,   # モデルとデータを配置するデバイス (CPUやGPU)
) -> tuple[list[float], list[tuple[int, dict[str, torch.Tensor]]]]:
    
    # モデルとデータを指定されたデバイス(CPU/GPU)に移動する
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    criterion = nn.MSELoss()    # 損失関数として平均二乗誤差を使用
    optimizer = torch.optim.AdamW(   # 最適化アルゴリズムとしてAdamWを使用し、モデルのパラメータを更新するためのオプティマイザを作成
        model.parameters(),     # モデルのパラメータをオプティマイザに渡す
        lr=config.learning_rate,    # 学習率を設定
        weight_decay=config.weight_decay,   # 重み減衰 (L2正則化) を設定
    )

    losses: list[float] = []    # 各エポックの損失を格納するリスト
    snapshots: list[tuple[int, dict[str, torch.Tensor]]] = []    # 後で動画を作るためのスナップショットを格納するリスト

    # 学習ループを実行する。指定されたエポック数だけ繰り返す。
    for epoch in range(1, config.epochs + 1):
        model.train()   # モデルを訓練モードに設定する
        optimizer.zero_grad()   # 勾配を初期化する（絶対に忘れないように）
        prediction = model(x)   # モデルを呼び出して、入力xに対する予測を取得する
        loss = criterion(prediction, y)  # 予測と正解yを比較して損失を計算する
        loss.backward()  # 勾配を計算する（逆伝播）
        optimizer.step()    # オプティマイザを呼び出して、モデルのパラメータを更新する

        loss_value = float(loss.item()) # 損失の値をPythonのfloat型に変換して取得する
        losses.append(loss_value)   # 計算した損失をリストに追加する

        # 指定されたエポック数ごとに、モデルのスナップショットを保存する。後で動画を作るために使用する。
        if epoch == 1 or epoch % config.snapshot_every == 0 or epoch == config.epochs:
            snapshots.append((epoch, copy.deepcopy(model.state_dict())))    # モデルの状態辞書をコピーして、スナップショットとして保存する。

        # 指定されたエポック数ごとに、現在のエポック数と損失をコンソールに出力する。
        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            print(f"epoch={epoch:04d} loss={loss_value:.6f}")

    return losses, snapshots


# 学習のチェックポイントを保存する関数。モデルの状態、損失の履歴、設定を指定されたパスに保存する。
def save_checkpoint(
    path: Path,
    model: nn.Module,
    losses: list[float],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # チェックポイントを保存するディレクトリが存在しない場合は作成する
    # torch.saveを使用して、モデルの状態辞書、損失の履歴、設定を指定されたパスに保存する。これにより、後で学習を再開したり、モデルを評価したりすることができる。
    torch.save(
        {
            "model_state_dict": model.state_dict(), # モデルの状態辞書を保存する (モデルのパラメータの値を含む)
            "losses": losses, # 損失の履歴を保存する
            "config": config, # 設定を保存する
        },
        path,   # 指定されたパスに保存する
    )


# 学習の進行をアニメーションとして保存する関数。モデルのスナップショットを使用して、学習の進行を可視化するアニメーションを作成し、指定されたパスに保存する。
def save_training_animation(
    path: Path, # アニメーションを保存するパス
    model: nn.Module,   # 学習させたモデル
    snapshots: list[tuple[int, dict[str, torch.Tensor]]],   # 学習の進行を可視化するためのモデルのスナップショットのリスト (エポック数とモデルの状態辞書のタプルのリスト)
    train_x: torch.Tensor,   # 学習データの入力 (x)
    train_y: torch.Tensor,   # 学習データの出力 (y)
    function_config: Any,    # 関数の設定
    animation_config: AnimationConfig,   # アニメーションの設定
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)  # アニメーションを保存するディレクトリが存在しない場合は作成する
    plot_x = torch.linspace(function_config.x_min, function_config.x_max, 400).unsqueeze(1) # 関数の定義域を400点で等間隔に分割したテンソルを作成し、列ベクトルの形状に変換する (400サンプル、1次元)
    target_y = target_function(plot_x, function_config) # 定義した関数を呼び出して、plot_xに対するターゲット値（正解の値）を計算する (400サンプル、1次元)

    # Matplotlibを使用して、学習の進行を可視化するためのグラフを作成する。学習データの散布図、ターゲット関数の曲線、モデルの予測の曲線を描画する。
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(train_x.squeeze().numpy(), train_y.squeeze().numpy(), s=14, alpha=0.45, label="noisy samples")
    ax.plot(plot_x.squeeze().numpy(), target_y.squeeze().numpy(), color="tab:blue", label="target function")
    prediction_line, = ax.plot([], [], color="tab:red", linewidth=2, label="model prediction")
    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 学習データとターゲット関数の両方を考慮して、y軸の範囲を設定する。学習データとターゲット関数の最小値と最大値を比較して、適切な範囲を決定する。
    y_min = min(float(train_y.min()), float(target_y.min())) - 1.0
    y_max = max(float(train_y.max()), float(target_y.max())) + 1.0
    ax.set_xlim(function_config.x_min, function_config.x_max)
    ax.set_ylim(y_min, y_max)

    device = next(model.parameters()).device  # モデルのパラメータが存在するデバイスを取得する (CPU/GPU)

    # アニメーションの各フレームを更新するための関数。フレームごとにモデルの状態を更新し、モデルの予測を計算して、グラフを更新する。
    def update(frame_index: int) -> tuple[Any, Any]:
        epoch, state_dict = snapshots[frame_index]  # スナップショットからエポック数とモデルの状態辞書を取得する
        model.load_state_dict(state_dict)   # モデルの状態をスナップショットの状態辞書に更新する
        model.eval()    # モデルを評価モードに設定する
        with torch.no_grad():   # 勾配計算を無効にするコンテキストマネージャー。これにより、予測の計算が高速化され、メモリ使用量も削減される。
            prediction = model(plot_x.to(device)).cpu()  # モデルを呼び出して、plot_xに対する予測を計算する。予測はモデルのデバイスで計算されるため、CPUに移動してからグラフに描画するためにcpu()を呼び出す。
        prediction_line.set_data(plot_x.squeeze().numpy(), prediction.squeeze().numpy())    # モデルの予測をグラフの予測ラインに設定する
        title.set_text(f"Training Progress - Epoch {epoch}")    # グラフのタイトルを更新して、現在のエポック数を表示する
        return prediction_line, title   # 更新された予測ラインとタイトルを返す (FuncAnimationがこれらを使用してグラフを更新するため)

    # FuncAnimationを使用して、学習の進行を可視化するアニメーションを作成する。update関数をフレームごとに呼び出して、グラフを更新する。
    animation = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / animation_config.fps, blit=False)

    # アニメーションを保存するためのライターを選択する。
    if path.suffix.lower() == ".mp4":
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=animation_config.fps)
    else:
        writer = PillowWriter(fps=animation_config.fps)

    # アニメーションを指定されたパスに保存する。FFMpegWriterやPillowWriterを使用して、アニメーションを動画ファイルやGIFファイルとして保存する。
    animation.save(path, writer=writer, dpi=animation_config.dpi)
    plt.close(fig)


def run(config_path: Path) -> None:
    config = load_config(config_path)   # 設定ファイルを読み込む
    set_seed(int(config["seed"]))   # 乱数シードを設定することで、学習の再現性を確保する

    device = torch.device(config.get("device", "cpu"))   # 設定ファイルからデバイスを取得する。指定がない場合はCPUを使用する。
    
    # 設定ファイルから関数の設定、学習の設定、アニメーションの設定を構築する。
    function_config = build_function_config(config["data"])
    training_config = build_training_config(config)
    animation_config = build_animation_config(config)
    paths = build_paths(config)

    x, y = generate_dataset(function_config, seed=int(config["seed"]))   # データセットを生成する
    model = build_model(config["model"])   # モデルを構築する

    losses, snapshots = train_model(model, x, y, training_config, device)   # モデルを学習する
    save_checkpoint(paths.checkpoint_path, model, losses, config)   # チェックポイントを保存する
    save_training_animation(paths.animation_path, model, snapshots, x, y, function_config, animation_config)   # 学習の進行を可視化するアニメーションを保存する

    # 学習の結果をコンソールに出力する。保存したチェックポイントとアニメーションのパスを表示する。
    print(f"saved checkpoint: {paths.checkpoint_path}")
    print(f"saved animation: {paths.animation_path}")


# メイン関数。ここでは、run関数を呼び出して、学習を実行する。
def main() -> None:
    run(CONFIG_PATH)

# スクリプトが直接実行されたときにmain関数を呼び出す
if __name__ == "__main__":
    main()
