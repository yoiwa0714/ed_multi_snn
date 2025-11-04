#!/bin/bash
# ed_multi_lif_snn.py 実行ラッパー（メインファイル用）
# nohup + バックグラウンド実行を自動化

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="/tmp/ed_main_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 ED-SNN Main 実行開始..."
echo "📝 ログファイル: $LOGFILE"

cd "$SCRIPT_DIR"
nohup python ed_multi_lif_snn.py "$@" > "$LOGFILE" 2>&1 &
PID=$!

echo "✅ プロセスID: $PID"
echo "📊 進捗確認: tail -f $LOGFILE"
echo "🔍 プロセス確認: ps aux | grep $PID"
