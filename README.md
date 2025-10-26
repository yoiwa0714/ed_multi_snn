# ED-SNN 公開版 - スパイキングニューラルネットワークのための純粋ED法実装

金子勇氏オリジナルError-Diffusion（ED）法をスパイキングニューラルネットワーク（SNN）に適用した実装です。

## 概要

本実装は、誤差逆伝播法を使用せず、生物学的に妥当な学習アルゴリズムでMNISTとFashion-MNISTの画像分類を実現します。

## 主要な特徴

- ✅ **純粋ED法**: 誤差逆伝播法・連鎖律を使用しない生物学的学習
- ✅ **全層LIF化**: 1706個の全ニューロンがLeaky Integrate-and-Fire（LIF）モデル
- ✅ **スパイク符号化**: ポアソン符号化による入力層のスパイク生成（150Hz, 50ms）
- ✅ **E/Iペア構造**: 興奮性・抑制性ニューロンペアによる生物学的妥当性
- ✅ **Dale's Principle**: ニューロンの重み符号保持（興奮性≥0、抑制性≤0）

## 達成正答率

- **MNIST**: テスト正答率85.0%（訓練正答率85.9%、1000サンプル×10エポック）
- **Fashion-MNIST**: テスト正答率82.0%（単層128構成）
- 汎化ギャップ: 0.9%（過学習なし、極めて良好）

## ファイル構成

```
ed_snn/
├── README.md                    # このファイル
├── ed_snn_mnist_published.py    # メインプログラム（MNIST/Fashion-MNIST専用、CIFAR削除版）
├── ed_snn_v025.py               # 元の完全版（参考用、CIFAR対応）
└── modules/                     # 必要なモジュール群
    ├── data_loader.py           # データローダー
    ├── accuracy_loss_verifier.py # 正答率・誤差検証
    ├── snn/                     # SNNモジュール
    │   ├── __init__.py
    │   ├── lif_neuron.py        # LIFニューロン実装
    │   └── spike_encoding.py    # スパイク符号化
    └── (その他必要なモジュール)
```

## 基本的な使い方

### MNIST学習（デフォルト設定）

```bash
python ed_snn_mnist_published.py --mnist --train 1000 --test 100 --epochs 10
```

### Fashion-MNIST学習

```bash
python ed_snn_mnist_published.py --fashion --train 1000 --test 100 --epochs 10
```

### リアルタイム可視化付き

```bash
python ed_snn_mnist_published.py --mnist --train 1000 --test 100 --epochs 10 --viz --heatmap
```

### スパイク符号化パラメータのカスタマイズ

```bash
python ed_snn_mnist_published.py --mnist --train 1000 --test 100 --epochs 10 \
  --use_input_lif --spike_encoding poisson \
  --spike_max_rate 150 --spike_sim_time 50 --spike_dt 1.0
```

## 主要なコマンドライン引数

### データセット

- `--mnist`: MNISTデータセット使用
- `--fashion`: Fashion-MNISTデータセット使用

### 学習設定

- `--train N`: 訓練サンプル数（デフォルト: 512）
- `--test N`: テストサンプル数（デフォルト: 512）
- `--epochs N`: エポック数（デフォルト: 10）
- `--hidden N1,N2,...`: 隠れ層構造（デフォルト: 128）
- `--batch N`: ミニバッチサイズ（デフォルト: 128）

### ED法パラメータ

- `--lr FLOAT`: 学習率（デフォルト: 0.1）
- `--ami FLOAT`: 初期アミン濃度（デフォルト: 0.25）
- `--dif FLOAT`: アミン拡散係数（デフォルト: 0.5）
- `--sig FLOAT`: シグモイド閾値（デフォルト: 1.2）

### LIFニューロンパラメータ

- `--v_rest FLOAT`: 静止膜電位（デフォルト: -65.0 mV）
- `--v_threshold FLOAT`: 発火閾値（デフォルト: -60.0 mV）
- `--v_reset FLOAT`: リセット電位（デフォルト: -70.0 mV）
- `--tau_m FLOAT`: 膜時定数（デフォルト: 20.0 ms）
- `--tau_ref FLOAT`: 不応期（デフォルト: 2.0 ms）

### スパイク符号化パラメータ

- `--use_input_lif`: 入力層のLIF化を有効化
- `--spike_encoding TYPE`: 符号化方式（poisson/rate/temporal、デフォルト: poisson）
- `--spike_max_rate FLOAT`: 最大発火率（デフォルト: 150.0 Hz）
- `--spike_sim_time FLOAT`: シミュレーション時間（デフォルト: 50.0 ms）
- `--spike_dt FLOAT`: 時間ステップ（デフォルト: 1.0 ms）

### 可視化

- `--viz`: リアルタイム学習進捗表示
- `--heatmap`: スパイク活動ヒートマップ表示
- `--save_fig DIR`: 図表保存ディレクトリ指定

### その他

- `--cpu`: CPU強制実行（GPU環境でも）
- `--no_shuffle`: データシャッフル無効化

## 必要な環境

- Python 3.8以上
- NumPy
- TensorFlow（データセット読み込み用）
- Matplotlib（可視化用）
- tqdm（進捗表示用）
- オプション: CuPy（GPU高速化用）

## ed_multi_snn.prompt.md完全準拠

本実装は以下の7つの拡張機能を完全実装しています：

1. **E/Iペア構造**: 各ピクセル→興奮性+抑制性ニューロン
2. **Dale's Principle**: 興奮性重み≥0、抑制性重み≤0の保持
3. **独立出力ニューロン**: クラスごとに独立した出力ニューロン
4. **アミン拡散学習**: 純粋ED法（誤差逆伝播なし）
5. **スパイク符号化**: ポアソン符号化（最適パラメータ: 150Hz, 50ms）
6. **LIFニューロン**: 全1706ニューロンがLIF化
7. **GPU計算支援**: CPU/GPU透明な切り替え

## 開発履歴

- v019: 金子勇氏オリジナルED法完全準拠達成（正答率76.4%）
- v020: リアルタイム可視化システム完成
- v021-v023: GPU最適化、過学習問題修正
- v024: GPU/CPU強制実行オプション対応
- v025: 全LIF化完成、スパイク符号化最適化、目標正答率達成（85.0%）
- **公開版**: CIFAR関連コードを削除、MNIST/Fashion-MNIST専用に最適化

## 参考文献

- 金子勇 (1999): Error-Diffusion法の原論文
- ed_multi_snn.prompt.md: 本実装の設計指針
- LIF neuron model: Leaky Integrate-and-Fire ニューロンモデル

## ライセンス

Original ED method by Isamu Kaneko (1999)  
Multi-layer & ED Core Parameters & SNN extension (2025)  
Public Release Version (2025-10-24)

---

**作成日**: 2025年10月24日  
**Base**: ed_snn_v025.py（全LIF化完成版）
