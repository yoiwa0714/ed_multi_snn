# ED-Multi SNN - スパイキングニューラルネットワークのための純粋ED法実装

金子勇氏オリジナルError-Diffusion（ED）法をスパイキングニューラルネットワーク（SNN）に適用した実装です。

## 概要

本実装は、誤差逆伝播法を使用せず、生物学的に妥当な学習アルゴリズムでMNISTとFashion-MNISTの画像分類を実現します。全ニューロンがLeaky Integrate-and-Fire（LIF）モデルで動作する、完全なスパイキングニューラルネットワークです。

## 主要な特徴

- ✅ **純粋ED法**: 誤差逆伝播法・連鎖律を使用しない生物学的学習
- ✅ **全層LIF化**: すべてのニューロンがLeaky Integrate-and-Fire（LIF）モデル
- ✅ **スパイク符号化**: ポアソン符号化による入力層のスパイク生成
- ✅ **E/Iペア構造**: 興奮性・抑制性ニューロンペアによる生物学的妥当性
- ✅ **Dale's Principle**: ニューロンの重み符号保持（興奮性≥0、抑制性≤0）
- ✅ **多層対応**: 任意の隠れ層構造（単層・多層）をサポート
- ✅ **高速化**: NumPy行列演算による効率的な実装
- ✅ **GPU対応**: CuPy使用時の自動GPU計算

## 達成正答率

- **MNIST**: テスト正答率85.0%（訓練正答率85.9%、1000サンプル×10エポック）
- **Fashion-MNIST**: テスト正答率82.0%（単層128構成）
- 汎化ギャップ: 0.9%（過学習なし、極めて良好）

## ファイル構成

```
ed_multi_snn/
├── README.md                      # このファイル
├── README_simple.md               # サンプル版の説明
├── ed_multi_lif_snn.py            # メインプログラム（公開版）
├── ed_multi_lif_snn_simple.py     # 教育用サンプル版
└── modules/                       # 必要なモジュール群
    ├── __init__.py
    ├── data_loader.py             # データローダー
    ├── accuracy_loss_verifier.py  # 正答率・誤差検証
    ├── snn/                       # SNNモジュール
    │   ├── __init__.py
    │   ├── lif_neuron.py          # LIFニューロン実装
    │   ├── snn_network.py         # SNNネットワーク
    │   ├── snn_network_fast.py    # 高速版SNNネットワーク
    │   ├── snn_network_fast_v2.py # 最適化版SNNネットワーク
    │   └── ed_core_fast_v2.py     # ED法コア実装
    ├── ed_learning/               # ED法モジュール
    │   ├── __init__.py
    │   ├── ed_core.py             # ED法基本実装
    │   └── ed_core_fast.py        # 高速版ED法実装
    ├── data/                      # データ管理
    │   ├── __init__.py
    │   └── dataset_manager.py     # データセット管理
    ├── utils/                     # ユーティリティ
    │   ├── __init__.py
    │   ├── dataset_manager.py     # データセット管理
    │   ├── font_config.py         # フォント設定
    │   └── profiler.py            # プロファイラー
    ├── visualization/             # 可視化
    │   ├── __init__.py
    │   └── fashion_mnist_analysis.py # Fashion-MNIST分析
    ├── snn_heatmap_visualizer.py  # ヒートマップ可視化
    └── snn_heatmap_integration.py # ヒートマップ統合
```

## 基本的な使い方

### MNIST学習（デフォルト設定）

```bash
python ed_multi_lif_snn.py --mnist --train 1000 --test 100 --epochs 10
```

### Fashion-MNIST学習

```bash
python ed_multi_lif_snn.py --fashion --train 1000 --test 100 --epochs 10
```

### 多層構造での学習

```bash
python ed_multi_lif_snn.py --mnist --train 1000 --test 100 --epochs 10 --hidden 256,128,64
```

### リアルタイム可視化付き

```bash
python ed_multi_lif_snn.py --mnist --train 1000 --test 100 --epochs 10 --viz --heatmap
```

### GPU使用

```bash
python ed_multi_lif_snn.py --mnist --train 1000 --test 100 --epochs 10
# GPU自動検出・使用（CuPyインストール時）
```

### 図表保存付き学習

```bash
python ed_multi_lif_snn.py --mnist --train 1000 --test 100 --epochs 10 --viz --heatmap --save_fig results/
# 学習進捗グラフとヒートマップをresults/ディレクトリに保存
```

## 主要なコマンドライン引数

### データセット

- `--mnist`: MNISTデータセット使用（デフォルト）
- `--fashion`: Fashion-MNISTデータセット使用

### 学習設定

- `--train N`: 訓練サンプル数（デフォルト: 512）
- `--test N`: テストサンプル数（デフォルト: 512）
- `--epochs N`: エポック数（デフォルト: 10）
- `--hidden N1,N2,...`: 隠れ層構造（デフォルト: 128）
- `--batch N`: ミニバッチサイズ（デフォルト: 128）
- `--seed N`: ランダムシード（デフォルト: ランダム）
- `--no_shuffle`: データシャッフル無効化

### ED法ハイパーパラメータ

- `--lr FLOAT`: 学習率 (alpha) - ニューロンの学習強度を制御（デフォルト: 0.1）
- `--ami FLOAT`: アミン濃度 (beta) - 初期誤差信号の強度（デフォルト: 0.25）
- `--dif FLOAT`: 拡散係数 (u1) - アミン（誤差信号）の拡散率（デフォルト: 0.5）
- `--sig FLOAT`: シグモイド閾値 (u0) - 活性化関数の感度（デフォルト: 1.2）
- `--w1 FLOAT`: 重み初期値1 - 興奮性ニューロンの初期重み（デフォルト: 1.0）
- `--w2 FLOAT`: 重み初期値2 - 抑制性ニューロンの初期重み（デフォルト: 1.0）

### LIFニューロンパラメータ

- `--v_rest FLOAT`: 静止膜電位（デフォルト: -65.0 mV）
- `--v_threshold FLOAT`: 発火閾値（デフォルト: -50.0 mV）
- `--v_reset FLOAT`: リセット電位（デフォルト: -65.0 mV）
- `--tau_m FLOAT`: 膜時定数（デフォルト: 10.0 ms）
- `--tau_ref FLOAT`: 不応期（デフォルト: 2.0 ms）
- `--simulation_time FLOAT`: シミュレーション時間（デフォルト: 50.0 ms）
- `--dt FLOAT`: 時間ステップ（デフォルト: 1.0 ms）
- `--R_m FLOAT`: 膜抵抗（デフォルト: 10.0 MΩ）
- `--spike_encoding METHOD`: スパイク符号化方法（デフォルト: poisson）
- `--spike_max_rate FLOAT`: 最大発火率 Hz（デフォルト: 100.0）
- `--spike_sim_time FLOAT`: スパイクシミュレーション時間 ms（デフォルト: 50.0）
- `--spike_dt FLOAT`: スパイク時間刻み ms（デフォルト: 1.0）

### 可視化

- `--viz`: リアルタイム学習進捗表示
- `--heatmap`: スパイク活動ヒートマップ表示
- `--save_fig DIR`: 図表保存ディレクトリ指定

### その他

- `--cpu`: CPU強制実行（GPU環境でも）
- `--verbose`: 詳細ログ表示
- `--verify_acc_loss`: 精度・誤差の検証レポートを表示

## 必要な環境

### 必須

- Python 3.8以上
- NumPy
- TensorFlow（データセット読み込み用）
- Matplotlib（可視化用）
- Seaborn（ヒートマップ可視化用）
- tqdm（進捗表示用）

### オプション

- CuPy（GPU高速化用）
- NVIDIA GPU + CUDA（GPU使用時）

## インストール

```bash
# 基本パッケージ
pip install numpy tensorflow matplotlib seaborn tqdm

# GPU使用時（オプション）
pip install cupy-cuda11x  # CUDA 11.x用
# または
pip install cupy-cuda12x  # CUDA 12.x用
```

## ed_multi_snn.prompt.md完全準拠

本実装は金子勇氏のオリジナルED法理論を完全に保持しながら、以下の拡張機能を実装しています：

### 1. 純粋ED法の保持

- **重要**: 誤差逆伝播法・連鎖律を一切使用しない
- アミン拡散メカニズムによる生物学的学習
- 出力層中心のエラー拡散型重み更新

### 2. E/Iペア構造

- 各入力ピクセル → 興奮性(+1) + 抑制性(-1) ニューロンペア
- MNIST: 784ピクセル → 1568ニューロン（784ペア）
- 生物学的妥当性の保証

### 3. Dale's Principle（デールの原理）

- 同種間結合: 正の重み制約
- 異種間結合: 負の重み制約
- 重み符号制約: `w *= ow[source] * ow[target]`

### 4. 独立出力ニューロンアーキテクチャ

- 各クラスが完全に独立した重み空間を保持
- 3次元重み配列: `w_ot_ot[出力ニューロン][送信先][送信元]`

### 5. 全層LIF化

- 入力層・隠れ層・出力層のすべてがLIFニューロン
- スパイク駆動型の完全なSNN実装
- 生物学的リアリズムの追求

### 6. 多層ニューラルネットワーク対応

- オリジナル仕様: 単一隠れ層
- 拡張機能: 複数隠れ層を自由に組み合わせ可能
- アミン拡散係数u1を多層間に適用

### 7. 高速化・GPU対応

- NumPy行列演算による並列計算
- CuPy統合による透明なGPU処理
- CPU/GPU自動切り替え

## 開発履歴

- **v019**: 金子勇氏オリジナルED法完全準拠達成
- **v020**: リアルタイム可視化システム完成
- **v021-v023**: GPU最適化、過学習問題修正
- **v024**: GPU/CPU強制実行オプション対応
- **v025**: 全LIF化完成、スパイク符号化最適化、目標正答率達成（85.0%）
- **公開版（2025-10-26）**:
  - 全層LIF化完成版
  - 未実装オプション削除
  - MNIST/Fashion-MNIST専用に最適化
  - ed_multi_snn.prompt.md 100%準拠

## 技術的詳細

### LIFニューロンモデル

膜電位の時間発展:

```
dV/dt = (V_rest - V + I_syn) / τ_m
```

- V: 膜電位
- V_rest: 静止膜電位
- I_syn: シナプス電流
- τ_m: 膜時定数

発火条件: `V ≥ V_threshold` → スパイク発火 → `V = V_reset`

### ED法学習則

重み更新:

```
Δw = α × amine × input × output_error
```

- α: 学習率
- amine: アミン濃度（誤差信号強度）
- input: 入力ニューロン活性
- output_error: 出力誤差

アミン拡散:

```
amine_hidden = u1 × amine_output
```

- u1: 拡散係数（0-1の範囲）

## 参考文献

- 金子勇 (1999): Error-Diffusion法の原論文
- ed_multi_snn.prompt.md: 本実装の設計指針・仕様書
- LIF neuron model: Leaky Integrate-and-Fire ニューロンモデル
- Dale's Principle: 神経伝達物質の単一性原理

## ライセンス

Original ED method by Isamu Kaneko (1999)  
Multi-layer & SNN extension implementation (2025)  
Public Release Version (2025-10-26)

---

**リポジトリ**: <https://github.com/yoiwa0714/ed_multi_snn>  
**作成日**: 2025年10月26日  
**Base**: ed_snn_v025.py（全LIF化完成版）

## サポート

問題や質問がある場合は、GitHubのIssuesでお知らせください。
