# 推奨コマンド集

## プロジェクト実行コマンド

### メイン実装（ed_multi_lif_snn.py）
```bash
# 直接実行
python ed_multi_lif_snn.py [options]

# バックグラウンド実行（推奨）
./run_ed_main.sh [options]

# 実行例
./run_ed_main.sh --dataset mnist --hidden_sizes 512 256 128 64 --epochs 20 --learning_rate 0.1
```

### シンプル実装（ed_multi_lif_snn_simple.py）
```bash
# 直接実行
python ed_multi_lif_snn_simple.py [options]

# バックグラウンド実行（推奨）
./run_ed_simple.sh [options]
```

### FReLU実装（ed_multi_frelu_snn.py）
```bash
python ed_multi_frelu_snn.py [options]
```

## 開発・デバッグコマンド

### ログ監視
```bash
# リアルタイムログ確認
tail -f /tmp/ed_main_YYYYMMDD_HHMMSS.log
tail -f /tmp/ed_simple_YYYYMMDD_HHMMSS.log

# プロセス確認
ps aux | grep ed_multi
```

### Git操作
```bash
git status
git add .
git commit -m "description"
git push origin main
```

### ファイル操作
```bash
# ディレクトリ確認
ls -la
find . -name "*.py" -type f

# モジュール構造確認
tree modules/
```

## システムコマンド（Linux環境）
```bash
# システム情報
uname -a
df -h
free -h

# Python環境
python --version
pip list

# プロセス管理
ps aux | grep python
kill -TERM <PID>
```