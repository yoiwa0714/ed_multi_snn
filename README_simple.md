# ED-Multi SNN Simple - 教育用サンプル版

`ed_multi_lif_snn_simple.py`は、ED-Multi SNNの学習プロセスを理解しやすくした教育用サンプル版です。

## 概要

このサンプル版は、メインプログラム（`ed_multi_lif_snn.py`）のコア機能を保ちながら、学習者が理解しやすいようにシンプル化されています。ED法の基本原理とSNNの動作を学ぶのに最適です。

## メインプログラムとの違い

### Simple版の特徴

- ✅ **シンプルな構造**: コアED法とLIF実装に焦点
- ✅ **詳細なコメント**: 各処理の説明が充実
- ✅ **基本機能のみ**: 必要最小限の機能で動作
- ✅ **学習向き**: ED法とSNNの理解に最適

### メイン版との主な違い

| 機能 | Simple版 | メイン版 |
|------|----------|----------|
| 対応データセット | MNIST, Fashion-MNIST | MNIST, Fashion-MNIST |
| 全層LIF化 | ✅ 対応 | ✅ 対応 |
| E/Iペア構造 | ✅ 対応 | ✅ 対応 |
| Dale's Principle | ✅ 対応 | ✅ 対応 |
| 多層構造 | ✅ 対応 | ✅ 対応 |
| リアルタイム可視化 | ✅ 基本機能 | ✅ 高度な可視化 |
| ヒートマップ | ✅ 基本機能 | ✅ 詳細な分析 |
| GPU対応 | ✅ 対応 | ✅ 対応 |
| プロファイリング | ❌ なし | ✅ 詳細な性能分析 |
| 詳細ログ | ❌ 基本のみ | ✅ 詳細ログ |
| コード行数 | 2,397行 | 2,764行 |

## 基本的な使い方

### MNIST学習

```bash
python ed_multi_lif_snn_simple.py --mnist --train 1000 --test 100 --epochs 10
```

### Fashion-MNIST学習

```bash
python ed_multi_lif_snn_simple.py --fashion --train 1000 --test 100 --epochs 10
```

### 可視化付き学習

```bash
python ed_multi_lif_snn_simple.py --mnist --train 1000 --test 100 --epochs 10 --viz --heatmap
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

### ED法ハイパーパラメータ

- `--lr FLOAT`: 学習率（デフォルト: 0.1）
- `--ami FLOAT`: アミン濃度（デフォルト: 0.25）
- `--dif FLOAT`: 拡散係数（デフォルト: 0.5）
- `--sig FLOAT`: シグモイド閾値（デフォルト: 1.2）
- `--w1 FLOAT`: 重み初期値1（デフォルト: 1.0）
- `--w2 FLOAT`: 重み初期値2（デフォルト: 1.0）

### LIFニューロンパラメータ

- `--v_rest FLOAT`: 静止膜電位（デフォルト: -65.0 mV）
- `--v_threshold FLOAT`: 発火閾値（デフォルト: -50.0 mV）
- `--v_reset FLOAT`: リセット電位（デフォルト: -65.0 mV）
- `--tau_m FLOAT`: 膜時定数（デフォルト: 10.0 ms）
- `--tau_ref FLOAT`: 不応期（デフォルト: 2.0 ms）

### 可視化

- `--viz`: リアルタイム学習進捗表示
- `--heatmap`: スパイク活動ヒートマップ表示

### その他

- `--cpu`: CPU強制実行
- `--no_shuffle`: データシャッフル無効化

## 学習のポイント

### 1. ED法の基本原理

```python
# 重み更新の基本式
Δw = α × amine × input × output_error
```

- **学習率（α）**: 学習の速さを制御
- **アミン濃度**: 誤差信号の強度
- **入力活性**: 入力ニューロンの活動
- **出力誤差**: 目標との差

### 2. アミン拡散メカニズム

```python
# 出力層から隠れ層への誤差拡散
amine_hidden = u1 × amine_output
```

- 誤差逆伝播を使わない生物学的な学習
- アミン（神経伝達物質）のような拡散

### 3. LIFニューロンの動作

```python
# 膜電位の時間発展
dV/dt = (V_rest - V + I_syn) / τ_m

# 発火条件
if V ≥ V_threshold:
    spike = True
    V = V_reset
```

### 4. E/Iペア構造

- 各ピクセル → 興奮性 + 抑制性ニューロンペア
- MNIST: 784ピクセル → 1568入力ニューロン

### 5. Dale's Principle

```python
# 重み符号制約
w *= ow[source] * ow[target]
```

- 同種間: 正の重み
- 異種間: 負の重み

## コードの理解を深めるために

### 推奨学習順序

1. **基本実行**: まずデフォルト設定で実行
   ```bash
   python ed_multi_lif_snn_simple.py --mnist --train 100 --test 10 --epochs 3
   ```

2. **可視化確認**: 学習過程を可視化
   ```bash
   python ed_multi_lif_snn_simple.py --mnist --train 100 --test 10 --epochs 3 --viz
   ```

3. **パラメータ変更**: 各パラメータの影響を確認
   ```bash
   # 学習率を変えてみる
   python ed_multi_lif_snn_simple.py --mnist --train 100 --test 10 --epochs 3 --lr 0.05
   
   # アミン拡散係数を変えてみる
   python ed_multi_lif_snn_simple.py --mnist --train 100 --test 10 --epochs 3 --dif 0.3
   ```

4. **ネットワーク構造変更**: 多層構造を試す
   ```bash
   python ed_multi_lif_snn_simple.py --mnist --train 100 --test 10 --epochs 3 --hidden 64,32
   ```

### コード内の重要セクション

1. **データ前処理**: E/Iペア化の実装
2. **LIFニューロン**: スパイク発火の実装
3. **ED法学習**: 重み更新の実装
4. **Dale's Principle**: 重み符号制約の実装

## 達成目標

- **理解目標**: ED法の基本原理とSNNの動作を理解
- **実装目標**: 簡単な修正や拡張ができる
- **性能目標**: MNIST 80%以上の正答率

## メイン版へのステップアップ

Simple版で基本を理解したら、メイン版（`ed_multi_lif_snn.py`）で以下を学習：

1. **高度な可視化**: 詳細なヒートマップと分析
2. **性能最適化**: GPU活用と高速化技術
3. **プロファイリング**: ボトルネック分析
4. **詳細ログ**: デバッグとチューニング

## 必要な環境

### 必須

- Python 3.8以上
- NumPy
- TensorFlow（データセット読み込み用）
- Matplotlib（可視化用）
- Seaborn（ヒートマップ用）
- tqdm（進捗表示用）

### オプション

- CuPy（GPU高速化用）

## インストール

```bash
# 基本パッケージ
pip install numpy tensorflow matplotlib seaborn tqdm

# GPU使用時（オプション）
pip install cupy-cuda11x  # CUDA 11.x用
```

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   ```bash
   # サンプル数を減らす
   python ed_multi_lif_snn_simple.py --mnist --train 100 --test 10 --epochs 3
   ```

2. **学習が進まない**
   ```bash
   # 学習率を調整
   python ed_multi_lif_snn_simple.py --mnist --train 100 --test 10 --epochs 3 --lr 0.05
   ```

3. **GPU使用時のエラー**
   ```bash
   # CPUで実行
   python ed_multi_lif_snn_simple.py --mnist --train 100 --test 10 --epochs 3 --cpu
   ```

## 参考資料

- **README.md**: メイン版の詳細説明
- **ed_multi_snn.prompt.md**: 実装の設計指針
- **modules/**: 各モジュールの実装

## ライセンス

Original ED method by Isamu Kaneko (1999)  
Educational sample implementation (2025)  
Public Release Version (2025-10-26)

---

**リポジトリ**: https://github.com/yoiwa0714/ed_multi_snn  
**作成日**: 2025年10月26日

## サポート

問題や質問がある場合は、GitHubのIssuesでお知らせください。
