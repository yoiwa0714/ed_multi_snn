# ED法 SNN プロジェクト概要

## プロジェクトの目的
金子勇氏オリジナルのError-Diffusion（ED）法をスパイキングニューラルネットワーク（SNN）に適用した実装プロジェクト。生物学的妥当性の高い学習アルゴリズムによる画像分類を実現。

## 技術スタック
- **言語**: Python
- **ニューロンモデル**: LIFニューロン、FReLU
- **データセット**: MNIST、Fashion-MNIST
- **可視化**: matplotlib、リアルタイム学習進捗表示
- **依存関係**: NumPy、PyTorch、matplotlib

## 主要実装ファイル
1. `ed_multi_lif_snn.py` - 100% LIFニューロンで構成されるネットワーク（メインファイル）
2. `ed_multi_lif_snn_simple.py` - 基本機能のみの実装
3. `ed_multi_frelu_snn.py` - FReLU活性化関数を用いたネットワーク

## プロジェクト構造
```
├── ed_multi_*.py          # メイン実装ファイル
├── modules/               # モジュールライブラリ
│   ├── ed_learning/       # ED法コア実装
│   ├── snn/              # SNNネットワーク実装
│   ├── data/             # データ管理
│   ├── utils/            # ユーティリティ
│   └── visualization/     # 可視化機能
├── backup/               # バックアップファイル
├── test_results/         # 学習試験結果
├── viz_results/          # 可視化結果（内部用）
├── viz_results_for_public/ # 公開用可視化結果
├── docs/                 # ドキュメント
└── ed_multi_snn.prompt.md # プロジェクト仕様書
```

## ED法の特徴
- 微分の連鎖律を使用しない生物学的妥当な学習
- 並列計算による高速化
- 勾配消失問題の回避
- アミン拡散による局所学習