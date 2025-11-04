# タスク完了時の実行項目

## コード品質チェック
1. **構文チェック**: `python -m py_compile *.py`
2. **import文の確認**: 不要なimportを削除
3. **コメントの確認**: 日本語コメントの文字化けチェック

## テスト実行
```bash
# メイン実装の動作確認
python ed_multi_lif_snn.py --dataset mnist --epochs 1 --test_mode

# シンプル実装の動作確認  
python ed_multi_lif_snn_simple.py --dataset mnist --epochs 1 --test_mode

# FReLU実装の動作確認
python ed_multi_frelu_snn.py --dataset mnist --epochs 1 --test_mode
```

## ドキュメント更新
1. **README.md**: 新機能の説明を追加
2. **ed_multi_snn.prompt.md**: 仕様書への機能追記
3. **TECHNICAL_DOCS.md**: 技術詳細の更新

## Git操作
```bash
# 変更確認
git status
git diff

# コミット
git add .
git commit -m "feat: [機能名] - [簡潔な説明]"

# プッシュ（公開リポジトリへ）
git push origin main
```

## 結果検証
1. **学習結果**: 可視化結果の確認
2. **ログファイル**: エラーメッセージの確認
3. **メモリ使用量**: リソース使用状況の確認

## バックアップ作成
```bash
# 重要ファイルのバックアップ
cp ed_multi_lif_snn.py backup/ed_multi_lif_snn_backup_$(date +%Y%m%d_%H%M%S).py
```

## 仕様書更新チェック
- 新機能が`ed_multi_snn.prompt.md`に記載されているか確認
- 実装と仕様書の整合性チェック