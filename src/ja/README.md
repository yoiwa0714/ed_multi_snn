# ED-Multi SNN (日本語コメント版)

この`src/ja/`ディレクトリには、日本語のコメントが付いたED-Multi SNNの実装が含まれています。

## 実行方法

```bash
# このディレクトリに移動
cd src/ja

# MNISTデータセットで基本的な学習を実行
python ed_multi_lif_snn.py --mnist --epochs 10 --viz

# Fashion-MNISTデータセットでの学習（可視化あり）
python ed_multi_lif_snn.py --fashion --epochs 20 --viz --heatmap

# 多層ネットワークでの学習
python ed_multi_lif_snn.py --mnist --hidden 512,256,128 --epochs 30 --viz
```

## 提供ファイル

- `ed_multi_lif_snn.py` - 完全LIF版（推奨）
- `ed_multi_lif_snn_simple.py` - シンプル版（学習用）
- `ed_multi_frelu_snn.py` - FReLU版（実験用）
- `modules/` - 共通モジュール群

## 詳細情報

- 詳細な使用方法: [../../USAGE.md](../../USAGE.md)
- 技術仕様: [../../docs/ja/](../../docs/ja/)
- プロジェクト概要: [../../README.md](../../README.md)

## English Version

英語コメント版は`../en/`ディレクトリをご利用ください。