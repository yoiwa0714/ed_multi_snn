# コードスタイルと規約

## コーディング規約
- **言語**: Python 3.12+
- **エンコーディング**: UTF-8
- **改行コード**: LF（Unix形式）

## 命名規則
- **ファイル名**: snake_case（例: `ed_multi_lif_snn.py`）
- **クラス名**: PascalCase（例: `LIFNeuron`）
- **関数・変数名**: snake_case（例: `calculate_amine_diffusion`）
- **定数**: UPPER_SNAKE_CASE（例: `DEFAULT_LEARNING_RATE`）

## コメント規則
- **日本語コメント**: 主要機能の説明に使用
- **英語コメント**: 技術的詳細やアルゴリズム説明
- **docstring**: 関数・クラスの説明（日本語）

## インポート順序
1. 標準ライブラリ
2. サードパーティライブラリ（numpy, torch, matplotlib等）
3. プロジェクト内モジュール

## フォーマット
- **インデント**: スペース4つ
- **行長**: 推奨80文字、最大120文字
- **型ヒント**: 主要な関数で使用

## ED法固有の命名規則
- **アミン関連**: `amine_*`（例: `amine_diffusion`, `amine_concentration`）
- **誤差関連**: `error_*`（例: `error_signal`, `error_diffusion`）
- **重み更新**: `update_weights`, `weight_modification`
- **スパイク関連**: `spike_*`（例: `spike_train`, `spike_output`）

## ファイル構成パターン
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# インポート
# クラス定義
# メイン関数
# if __name__ == "__main__": ブロック
```