# ED-SNN 実行方法ガイド

## 簡単な実行方法（推奨）

### ラッパースクリプトを使用（nohup不要）

```bash
# 教育用サンプルコード（デフォルトで高正答率）
./run_ed_simple.sh

# 可視化付き実行
./run_ed_simple.sh --viz --heatmap

# パラメータカスタマイズ
./run_ed_simple.sh --train 1000 --test 100 --epochs 10 --viz

# メインファイル実行
./run_ed_main.sh --mnist --train 1000 --test 100
```

### ラッパースクリプトの利点

✅ `nohup`と`&`を毎回入力不要
✅ ログファイルが自動的にタイムスタンプ付きで生成
✅ プロセスIDと確認コマンドを自動表示
✅ Ctrl+C自動発生問題を完全回避

### 進捗確認方法

実行後に表示されるログファイルを確認:

```bash
# リアルタイムで進捗確認
tail -f /tmp/ed_simple_20251025_221234.log

# 最新の進捗確認
tail -30 /tmp/ed_simple_20251025_221234.log

# 正答率のみ抽出
tail -50 /tmp/ed_simple_20251025_221234.log | grep "訓精=\|テ精="
```

## 直接実行方法

### 通常実行（可視化なし）

```bash
python ed_multi_lif_snn_simple.py
python ed_multi_lif_snn.py --mnist --train 1000 --test 100
```

### 可視化付き実行（nohup使用）

```bash
# 教育用サンプル
nohup python ed_multi_lif_snn_simple.py --viz --heatmap > /tmp/ed_log.txt 2>&1 &

# メインファイル
nohup python ed_multi_lif_snn.py --mnist --viz --heatmap > /tmp/ed_log.txt 2>&1 &
```

## ファイル構成

```text
ed_multi_snn/
├── ed_multi_lif_snn.py              # 完全SNN実装 (LIF版)
├── ed_multi_lif_snn_simple.py       # 教育用サンプルコード (Simple版)
├── ed_multi_frelu_snn.py            # FReLU実験版
├── USAGE.md                         # このファイル
├── README.md                        # プロジェクト全体のREADME
├── TECHNICAL_DOCS.md                # 技術詳細ドキュメント
├── PERFORMANCE_REPORT.md            # パフォーマンス分析
├── test_results/                    # 学習結果レポート
│   ├── lif_snn_learning_test_results.md    # LIF版学習結果
│   └── frelu_snn_learning_test_results.md  # FReLU版学習結果
└── modules/                         # 必要最小限のモジュール
    ├── data_loader.py               # データローダー
    ├── accuracy_loss_verifier.py    # 正答率・誤差検証
    ├── snn_heatmap_integration.py   # ヒートマップ統合
    └── snn/                         # SNNコア実装
        └── lif_neuron.py            # LIFニューロン
```

## デフォルト設定（教育用サンプルコード）

`ed_multi_lif_snn_simple.py`は以下のパラメータがデフォルトで最適化済み:

- ✅ `enable_lif = True` (全層LIF化)
- ✅ `use_input_lif = True` (入力層スパイク符号化)
- ✅ `spike_max_rate = 100Hz` (最大発火率)
- ✅ `spike_encoding = poisson` (ポアソン符号化)
- ✅ `epochs = 10` (エポック数)
- ✅ `hidden = 128` (隠れ層)
- ✅ `batch = 128` (ミニバッチサイズ)

**オプション指定なしで78-81%の正答率を達成可能！**

## トラブルシューティング

### Ctrl+C自動発生問題

**解決済み**: ラッパースクリプト(`run_ed_*.sh`)を使用してください。

### プロセスの停止方法

```bash
# プロセスIDを確認
ps aux | grep "ed_multi_lif_snn"

# プロセスを停止
kill <PID>

# 強制停止（必要な場合のみ）
kill -9 <PID>
```

### ログファイルの場所

ラッパースクリプト使用時は実行時に表示されます:
```
📝 ログファイル: /tmp/ed_simple_20251025_221234.log
```

直接実行時は指定したパスに保存されます。

## 実行例

### 基本実行（最もシンプル）

```bash
./run_ed_simple.sh
```

### 可視化付き実行

```bash
./run_ed_simple.sh --viz --heatmap
```

### カスタムパラメータ実行

```bash
./run_ed_simple.sh --train 2000 --test 200 --epochs 20 --learning_rate 0.15
```

### メインファイルで大規模学習

```bash
./run_ed_main.sh --mnist --train 5000 --test 1000 --epochs 50 --viz
```

---

**最終更新**: 2025-10-25
**バージョン**: 1.0.0
