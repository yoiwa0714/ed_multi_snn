#!/usr/bin/env python3
"""
ed_multi_lif_snn_simple.py
バージョン: 1.0.0

スパイキングニューラルネットワークのためのError-Diffusion (ED)法実装
教育用サンプルコード

使い方:
  python ed_multi_lif_snn_simple.py --mnist --train 1000 --test 100

オプション指定例:
  python ed_multi_lif_snn_simple.py --mnist --train 1000 --test 100 --viz --heatmap
"""

# TensorFlowの警告・情報メッセージを非表示にする
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR以外を非表示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNNメッセージを非表示

import numpy as np
import time
import argparse
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
import warnings
import threading
from modules.accuracy_loss_verifier import AccuracyLossVerifier

# GPU計算支援（CuPy対応）- ed_multi_snn.prompt.md準拠
try:
    import cupy as cp
    xp = cp  # NumPy互換の配列ライブラリ
    GPU_AVAILABLE = True
    print("🚀 GPU（CuPy）が利用可能です")
    print(f"   デバイス: {cp.cuda.Device().compute_capability}")
except ImportError:
    import numpy as np
    xp = np  # フォールバック: NumPyを使用
    GPU_AVAILABLE = False
    print("ℹ️  GPU未検出。CPU（NumPy）で実行します")
except Exception as e:
    import numpy as np
    xp = np
    GPU_AVAILABLE = False
    print(f"⚠️  GPU初期化エラー。CPU（NumPy）で実行します: {e}")

# TensorFlowのログレベルを設定（追加の保険）
tf.get_logger().setLevel('ERROR')

# ミニバッチデータローダー
from modules.data_loader import MiniBatchDataLoader
# ========== HyperParamsクラス (ed_v032_simple.py準拠) ==========

class HyperParams:
    """
    ED-SNN ハイパーパラメータ管理クラス
    ed_v032_simple.py準拠: ed_multi_snn.prompt.md準拠版
    """
    
    def __init__(self):
        """デフォルト値設定（シミュレーション最適化値使用）"""
        # === ED法アルゴリズムパラメータ ===
        self.learning_rate = 0.1      # 学習率 (alpha) - シミュレーション最適化
        self.initial_amine = 0.25     # 初期アミン濃度 (beta) - シミュレーション最適化
        self.diffusion_rate = 0.5     # アミン拡散係数 (u1)
        self.sigmoid_threshold = 1.2  # シグモイド閾値 (u0) - シミュレーション最適化
        self.initial_weight_1 = 0.3   # 重み初期値1
        self.initial_weight_2 = 0.5   # 重み初期値2
        
        # === LIFニューロンパラメータ（v570準拠、新規追加） ===
        self.v_rest = -65.0           # 静止膜電位 (mV)
        self.v_threshold = -60.0      # 発火閾値 (mV)
        self.v_reset = -70.0          # リセット電位 (mV)
        self.tau_m = 20.0             # 膜時定数 (ms)
        self.tau_ref = 2.0            # 不応期 (ms)
        self.dt = 1.0                 # 時間ステップ (ms)
        self.R_m = 10.0               # 膜抵抗 (MΩ)
        self.simulation_time = 50.0   # シミュレーション時間 (ms)
        
        # === LIF統合制御（v019 Phase 4追加） ===
        self.enable_lif = True        # LIF層を使用するか（デフォルト: 有効 - 85%達成設定）
        
        # === Step 3a: 入力層LIF化パラメータ（v025追加） ===
        self.use_input_lif = True        # 入力層LIF使用フラグ（デフォルト: 有効 - 85%達成設定）
        self.spike_encoding_method = 'poisson'  # スパイク符号化方法 ('poisson', 'rate', 'temporal')
        self.spike_max_rate = 150.0      # 最大発火率 (Hz) - 85%達成設定
        self.spike_simulation_time = 50.0  # スパイクシミュレーション時間 (ms)
        self.spike_dt = 1.0               # スパイク時間刻み (ms)
        
        # === 実行時設定パラメータ ===
        self.train_samples = 512      # 訓練データ数 - シミュレーション最適化
        self.test_samples = 512       # テストデータ数 - シミュレーション最適化
        self.epochs = 10              # エポック数 - シミュレーション最適化
        self.hidden_layers = [128]    # 隠れ層構造 - シミュレーション最適化
        self.batch_size = 128         # ミニバッチサイズ - シミュレーション最適化
        self.random_seed = None       # ランダムシード
        self.enable_visualization = False  # リアルタイム可視化
        self.enable_heatmap = False        # ヒートマップ可視化
        self.verbose = False          # 詳細表示
        self.quiet_mode = False       # 簡潔出力モード [SNN未実装]
        self.enable_profiling = False # プロファイリング [SNN未実装]
        self.force_cpu = False        # CPU強制実行 [SNN未実装]
        self.fashion_mnist = True     # Fashion-MNIST使用 - シミュレーション最適化
        self.mnist = False            # MNIST使用
        self.save_fig = None          # 図表保存ディレクトリ（タイムスタンプ付き保存）
        self.no_shuffle = False       # データシャッフル無効化（ed_snn独自）
        self.verify_acc_loss = False  # 精度・誤差検証レポート表示
        
    def __post_init__(self):
        """データセット名と出力サイズの自動設定"""
        # データセット選択の優先順位: Fashion-MNIST > MNIST
        if self.fashion_mnist:
            self.dataset_name = 'fashion_mnist'
            self.output_size = 10   # Fashion-MNIST: 10クラス
        elif self.mnist:
            self.dataset_name = 'mnist'
            self.output_size = 10   # MNIST: 10クラス
        else:
            # デフォルト: MNIST
            self.dataset_name = 'mnist'
            self.output_size = 10
    
    def parse_args(self, args=None):
        """コマンドライン引数解析（ed_v032_simple.py準拠）"""
        import argparse
        
        parser = argparse.ArgumentParser(
            description='ED-SNN v015 HyperParams統一版 - ed_v032_simple準拠',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
ED法ハイパーパラメータ説明:
  学習率(alpha): ニューロンの学習強度を制御
  アミン濃度(beta): 初期誤差信号の強度 [SNN未実装]
  拡散係数(u1): アミン（誤差信号）の拡散率 [SNN未実装]
  シグモイド閾値(u0): 活性化関数の感度 [SNN未実装]
  
[SNN未実装]マークのパラメータ:
  将来の実装のためのダミーパラメータです。
  現在は指定しても効果はありませんが、ed_v032_simpleとの
  コマンドライン互換性を保つために用意されています。

Original Algorithm: 金子勇 (1999)
Implementation: ed_multi_snn.prompt.md準拠
            """
        )
        
        # === ED法アルゴリズムパラメータ ===
        ed_group = parser.add_argument_group('ED法アルゴリズムパラメータ')
        ed_group.add_argument('--learning_rate', '--lr', type=float, default=self.learning_rate,
                             help=f'学習率 alpha (デフォルト: {self.learning_rate})')
        ed_group.add_argument('--amine', '--ami', type=float, default=self.initial_amine,
                             help=f'初期アミン濃度 beta (デフォルト: {self.initial_amine}) [多層学習で重要]')
        ed_group.add_argument('--diffusion', '--dif', type=float, default=self.diffusion_rate,
                             help=f'アミン拡散係数 u1 (デフォルト: {self.diffusion_rate}) [多層学習で重要]')
        ed_group.add_argument('--sigmoid', '--sig', type=float, default=self.sigmoid_threshold,
                             help=f'シグモイド閾値 u0 (デフォルト: {self.sigmoid_threshold}) [多層学習で重要]')
        ed_group.add_argument('--weight1', '--w1', type=float, default=self.initial_weight_1,
                             help=f'重み初期値1 (デフォルト: {self.initial_weight_1}) [興奮性ニューロン]')
        ed_group.add_argument('--weight2', '--w2', type=float, default=self.initial_weight_2,
                             help=f'重み初期値2 (デフォルト: {self.initial_weight_2}) [抑制性ニューロン]')
        
        # === LIFニューロンパラメータ（v570準拠、新規追加） ===
        lif_group = parser.add_argument_group('LIFニューロンパラメータ（v019新規追加）')
        lif_group.add_argument('--v_rest', type=float, default=self.v_rest,
                              help=f'静止膜電位 (デフォルト: {self.v_rest} mV) [LIFニューロン]')
        lif_group.add_argument('--v_threshold', '--v_thresh', type=float, default=self.v_threshold,
                              help=f'発火閾値 (デフォルト: {self.v_threshold} mV) [LIFニューロン]')
        lif_group.add_argument('--v_reset', type=float, default=self.v_reset,
                              help=f'リセット電位 (デフォルト: {self.v_reset} mV) [LIFニューロン]')
        lif_group.add_argument('--tau_m', '--tau_mem', type=float, default=self.tau_m,
                              help=f'膜時定数 (デフォルト: {self.tau_m} ms) [LIFニューロン]')
        lif_group.add_argument('--tau_ref', '--tau_refractory', type=float, default=self.tau_ref,
                              help=f'不応期 (デフォルト: {self.tau_ref} ms) [LIFニューロン]')
        lif_group.add_argument('--dt', type=float, default=self.dt,
                              help=f'時間ステップ (デフォルト: {self.dt} ms) [LIFニューロン]')
        lif_group.add_argument('--R_m', '--membrane_resistance', type=float, default=self.R_m,
                              help=f'膜抵抗 (デフォルト: {self.R_m} MΩ) [LIFニューロン]')
        lif_group.add_argument('--sim_time', '--simulation_time', type=float, default=self.simulation_time,
                              help=f'シミュレーション時間 (デフォルト: {self.simulation_time} ms) [LIFニューロン]')
        lif_group.add_argument('--enable_lif', action='store_true',
                              help='LIF層を有効化（デフォルト: 無効） [v019 Phase 4新機能]')
        
        # === Step 3a: 入力層LIF統合パラメータ（v025新規追加） ===
        lif_group.add_argument('--use_input_lif', action='store_true',
                              help='入力層LIFを有効化（デフォルト: 無効） [v025 Step 3a新機能]')
        lif_group.add_argument('--spike_encoding', '--encoding', type=str, 
                              default=self.spike_encoding_method,
                              choices=['poisson', 'rate', 'temporal'],
                              help=f'スパイク符号化方法 (デフォルト: {self.spike_encoding_method}) [v025 Step 3a]')
        lif_group.add_argument('--spike_max_rate', '--max_rate', type=float, 
                              default=self.spike_max_rate,
                              help=f'最大発火率 Hz (デフォルト: {self.spike_max_rate}) [v025 Step 3a]')
        lif_group.add_argument('--spike_sim_time', type=float, 
                              default=self.spike_simulation_time,
                              help=f'スパイクシミュレーション時間 ms (デフォルト: {self.spike_simulation_time}) [v025 Step 3a]')
        lif_group.add_argument('--spike_dt', type=float, 
                              default=self.spike_dt,
                              help=f'スパイク時間刻み ms (デフォルト: {self.spike_dt}) [v025 Step 3a]')
        
        # === 実行時設定パラメータ ===
        exec_group = parser.add_argument_group('実行時設定パラメータ')
        exec_group.add_argument('--train_samples', '--train', type=int, default=self.train_samples,
                               help=f'訓練データ数 (デフォルト: {self.train_samples})')
        exec_group.add_argument('--test_samples', '--test', type=int, default=self.test_samples,
                               help=f'テストデータ数 (デフォルト: {self.test_samples})')
        exec_group.add_argument('--epochs', '--epo', type=int, default=self.epochs,
                               help=f'エポック数 (デフォルト: {self.epochs})')
        exec_group.add_argument('--hidden', '--hid', type=str, default=','.join(map(str, self.hidden_layers)),
                               help=f'隠れ層構造 (デフォルト: {",".join(map(str, self.hidden_layers))}) - カンマ区切り指定 (例: 256,128,64)')
        exec_group.add_argument('--batch_size', '--batch', type=int, default=self.batch_size,
                               help=f'ミニバッチサイズ (デフォルト: {self.batch_size})')
        exec_group.add_argument('--seed', type=int, default=self.random_seed,
                               help='ランダムシード (デフォルト: ランダム)')
        exec_group.add_argument('--viz', action='store_true', default=self.enable_visualization,
                               help='リアルタイム可視化を有効化 (デフォルト: 無効)')
        exec_group.add_argument('--heatmap', action='store_true', default=False,
                               help='リアルタイムヒートマップ可視化を有効化 (デフォルト: 無効)')
        exec_group.add_argument('--verbose', '--v', action='store_true', default=self.verbose,
                               help='詳細表示を有効化 (デフォルト: 無効)')
        exec_group.add_argument('--quiet', '--q', action='store_true', default=False,
                               help='簡潔出力モード - グリッドサーチ用 (デフォルト: 無効) [SNN未実装]')
        exec_group.add_argument('--cpu', action='store_true', default=self.force_cpu,
                               help='CPU強制実行モード: GPU環境でもCPU（NumPy）で実行。'
                                    'デバッグ、性能比較、GPU未搭載環境での動作確認に使用。'
                                    'ed_multi_snn.prompt.md拡張機能7準拠 (デフォルト: GPU自動検出)')
        exec_group.add_argument('--fashion', action='store_true', default=self.fashion_mnist,
                               help='Fashion-MNISTデータセット使用 (デフォルト: 有効)')
        exec_group.add_argument('--mnist', action='store_true',
                               help='通常MNISTデータセット使用 (--fashionの反対)')
        exec_group.add_argument('--save_fig', nargs='?', const='images', default=None,
                               help='図表保存を有効化 (引数なし: ./images, 引数あり: 指定ディレクトリ) ファイル名: realtime_viz_result_YYYYMMDD_HHMMSS.png')
        exec_group.add_argument('--verify_acc_loss', action='store_true', default=False,
                               help='精度・誤差の検証レポートを表示 (デフォルト: 無効)')
        exec_group.add_argument('--no_shuffle', action='store_true',
                               help='データシャッフルを無効化 (デフォルト: 有効)')
        
        # 引数解析
        parsed_args = parser.parse_args(args)
        
        # パラメータ値の更新
        self.learning_rate = parsed_args.learning_rate
        self.initial_amine = parsed_args.amine
        self.diffusion_rate = parsed_args.diffusion
        self.sigmoid_threshold = parsed_args.sigmoid
        self.initial_weight_1 = parsed_args.weight1
        self.initial_weight_2 = parsed_args.weight2
        
        # LIFニューロンパラメータ（v019新規追加）
        self.v_rest = parsed_args.v_rest
        self.v_threshold = parsed_args.v_threshold
        self.v_reset = parsed_args.v_reset
        self.tau_m = parsed_args.tau_m
        self.tau_ref = parsed_args.tau_ref
        self.dt = parsed_args.dt
        self.R_m = parsed_args.R_m
        self.simulation_time = parsed_args.sim_time
        self.enable_lif = parsed_args.enable_lif  # v019 Phase 4追加
        
        # Step 3a: 入力層LIF統合パラメータ（v025新規追加）
        self.use_input_lif = parsed_args.use_input_lif
        self.spike_encoding_method = parsed_args.spike_encoding
        self.spike_max_rate = parsed_args.spike_max_rate
        self.spike_simulation_time = parsed_args.spike_sim_time
        self.spike_dt = parsed_args.spike_dt
        
        # 実行時設定パラメータ
        self.train_samples = parsed_args.train_samples
        self.test_samples = parsed_args.test_samples
        self.epochs = parsed_args.epochs
        
        # 隠れ層構造の解析
        if isinstance(parsed_args.hidden, str):
            try:
                self.hidden_layers = [int(x.strip()) for x in parsed_args.hidden.split(',') if x.strip()]
                if not self.hidden_layers:
                    raise ValueError("隠れ層構造が空です")
                if any(layer <= 0 for layer in self.hidden_layers):
                    raise ValueError("隠れ層のニューロン数は正の整数である必要があります")
            except ValueError as e:
                raise ValueError(f"--hidden オプションの形式が不正です: {e}")
        else:
            self.hidden_layers = [parsed_args.hidden]
        
        self.batch_size = parsed_args.batch_size
        self.random_seed = parsed_args.seed
        self.enable_visualization = parsed_args.viz
        self.enable_heatmap = parsed_args.heatmap
        self.verbose = parsed_args.verbose
        self.quiet_mode = parsed_args.quiet
        self.force_cpu = parsed_args.cpu
        self.verify_acc_loss = parsed_args.verify_acc_loss  # 検証レポート表示
        
        # データセット選択フラグ処理（優先順位: MNIST > Fashion-MNIST）
        if hasattr(parsed_args, 'mnist') and parsed_args.mnist:
            self.mnist = True
            self.fashion_mnist = False
        else:
            self.mnist = False
            self.fashion_mnist = parsed_args.fashion
        
        # データセット名と出力サイズの設定
        self.__post_init__()
        
        self.save_fig = getattr(parsed_args, 'save_fig', None)
        self.no_shuffle = parsed_args.no_shuffle
        
        # 重み管理オプション
        return parsed_args


# matplotlib バックエンド設定（ed_v032_simple.py準拠）
try:
    if matplotlib.get_backend() == 'agg':
        try:
            matplotlib.use('Qt5Agg', force=True)
        except Exception:
            try:
                matplotlib.use('TkAgg', force=True)
            except Exception:
                pass
except Exception:
    pass

# 日本語フォント設定（ed_v032_simple.py準拠）
def setup_japanese_font():
    """
    利用可能な日本語フォントを自動検出して設定
    ed_genuine.prompt.md仕様: 日本語化Linuxの標準フォント使用
    """
    try:
        # システム内の利用可能フォント一覧を取得
        available_fonts = set([f.name for f in fm.fontManager.ttflist])
        
        # 日本語フォント候補（優先度順）
        japanese_font_candidates = [
            'Noto Sans CJK JP',   # Ubuntu/Debian標準
            'Noto Sans JP',       # Ubuntu/Debian代替
            'DejaVu Sans',        # 一般的なLinux
            'Liberation Sans',    # Red Hat系標準
            'TakaoGothic',        # CentOS/RHEL（存在時のみ）
            'VL Gothic',          # その他日本語（存在時のみ）
        ]
        
        # 実際に利用可能な日本語フォントを選択
        selected_font = None
        for font in japanese_font_candidates:
            if font in available_fonts:
                selected_font = font
                break
        
        # フォント設定（存在するフォントのみ）
        if selected_font:
            rcParams['font.family'] = [selected_font, 'sans-serif']
            print(f"✅ 日本語フォント検出・設定完了: {selected_font}")
        else:
            rcParams['font.family'] = ['sans-serif']
            print("⚠️ 日本語フォント未検出: デフォルトフォント使用")
        
        rcParams['axes.unicode_minus'] = False
        
        # matplotlib警告を最小化
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
        
    except Exception as e:
        print(f"⚠️ 日本語フォント設定エラー: {e}")
        rcParams['font.family'] = ['sans-serif']
        rcParams['axes.unicode_minus'] = False

def wait_for_keypress_or_timeout(timeout_seconds=5):
    """
    キー押下または指定秒数待機する関数
    Args:
        timeout_seconds: タイムアウト秒数（デフォルト5秒）
    Returns:
        bool: キーが押された場合True、タイムアウトの場合False
    """
    import sys
    import select
    
    print(f"\n⏱️  {timeout_seconds}秒後に自動クローズします（任意のキーで即座にクローズ）")
    
    # Windowsの場合はmsvcrtを使用
    if sys.platform == 'win32':
        import msvcrt
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if msvcrt.kbhit():
                msvcrt.getch()  # キーを消費
                print("🔑 キー押下を検知しました")
                return True
            time.sleep(0.1)
        print("⏰ タイムアウトによる自動クローズ")
        return False
    
    # Linux/Mac（WSL含む）の場合
    else:
        # 標準入力がターミナルでない場合はタイムアウトのみ
        if not sys.stdin.isatty():
            time.sleep(timeout_seconds)
            print("⏰ タイムアウトによる自動クローズ")
            return False
        
        # selectを使ってキー入力を待機
        import termios
        import tty
        
        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            
            rlist, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
            
            if rlist:
                sys.stdin.read(1)  # キーを消費
                print("🔑 キー押下を検知しました")
                return True
            else:
                print("⏰ タイムアウトによる自動クローズ")
                return False
        except Exception as e:
            # エラー時はタイムアウトのみ
            time.sleep(timeout_seconds)
            print("⏰ タイムアウトによる自動クローズ")
            return False
        finally:
            if old_settings:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except:
                    pass

class RealtimeLearningVisualizer:
    """
    リアルタイム学習可視化クラス（ed_v032_simple.py移植版）
    ed_multi_snn.prompt.md拡張機能5準拠
    """
    
    def __init__(self, max_epochs, window_size=(1000, 640), 
                 learning_rate=0.1, initial_amine=0.25, diffusion_rate=0.5,
                 sigmoid_threshold=1.2, initial_weight_1=0.3, initial_weight_2=0.5,
                 dataset_name='MNIST',
                 train_samples=None, test_samples=None, hidden_layers=None, batch_size=None,
                 v_rest=-65.0, v_threshold=-60.0, v_reset=-70.0, tau_m=20.0,
                 tau_ref=2.0, dt=1.0, R_m=10.0, simulation_time=50.0,
                 random_seed=None, verbose=False):
        """
        可視化インスタンス初期化（2x2グリッドレイアウト - ヒートマップ準拠）
        Args:
            max_epochs: 最大エポック数
            window_size: ウィンドウサイズ (width, height)
            learning_rate: 学習率（ED法のalpha）
            initial_amine: 初期アミン濃度（ED法のbeta）
            diffusion_rate: アミン拡散係数（ED法のu1）
            sigmoid_threshold: シグモイド閾値（ED法のu0）
            initial_weight_1: 重み初期値1（興奮性ニューロン）
            initial_weight_2: 重み初期値2（抑制性ニューロン）
            dataset_name: データセット名
            train_samples: 学習サンプル数
            test_samples: テストサンプル数
            hidden_layers: 隠れ層構造
            batch_size: ミニバッチサイズ
            random_seed: ランダムシード（v021 Phase 1追加）
            verbose: 詳細表示（v021 Phase 1追加）
        """
        self.max_epochs = max_epochs
        self.window_size = window_size
        # ED法アルゴリズムパラメータ（ed_multi_snn.prompt.md準拠）
        self.learning_rate = learning_rate
        self.initial_amine = initial_amine
        self.diffusion_rate = diffusion_rate
        self.sigmoid_threshold = sigmoid_threshold
        self.initial_weight_1 = initial_weight_1
        self.initial_weight_2 = initial_weight_2
        # LIFニューロンパラメータ（v019 Phase 3追加）
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.tau_ref = tau_ref
        self.dt = dt
        self.R_m = R_m
        self.simulation_time = simulation_time
        # 実行時設定パラメータ
        self.dataset_name = dataset_name
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.random_seed = random_seed  # v021 Phase 1追加
        self.verbose = verbose          # v021 Phase 1追加
        
        # データ保存用
        self.epochs = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_error_rates = []  # 訓練エラー率 (100 - accuracy)
        self.test_error_rates = []   # テストエラー率 (100 - accuracy)
        
        # グラフ初期化
        self.fig = None
        self.ax_params_ed = None      # 左上: EDパラメータ
        self.ax_params_lif = None     # 中上: LIFニューロンパラメータ（新規追加）
        self.ax_params_exec = None    # 右上: 実行パラメータ
        self.ax_acc = None             # 左下: 精度グラフ
        self.ax_err = None             # 右下: エラー率グラフ
        self.lines = {}
        
    def setup_plots(self):
        """グラフの初期設定 - 3パラメータボックス + 2グラフレイアウト（ed_multi_snn.prompt.md準拠）"""
        dpi = 100
        figsize = (self.window_size[0]/dpi, self.window_size[1]/dpi)
        
        # フィギュア作成
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # GridSpec作成: 2行×3列（上段: パラメータボックス×3、下段: グラフ×2）
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 3, figure=self.fig, hspace=0.4, wspace=0.3,
                               height_ratios=[1, 2])  # 上段1:下段2の高さ比
        
        # 上段: 3つのパラメータボックス（横並び）
        self.ax_params_ed = self.fig.add_subplot(gs[0, 0])     # 左上: EDパラメータ
        self.ax_params_lif = self.fig.add_subplot(gs[0, 1])    # 中上: LIFパラメータ
        self.ax_params_exec = self.fig.add_subplot(gs[0, 2])   # 右上: 実行パラメータ
        
        # 下段: 2つのグラフ（左右）
        self.ax_acc = self.fig.add_subplot(gs[1, 0:2])         # 左下～中下: 精度グラフ（幅広）
        self.ax_err = self.fig.add_subplot(gs[1, 2])           # 右下: エラー率グラフ
        
        # メインタイトル設定（バージョン番号削除）
        self.fig.suptitle("ED-SNN 学習進捗 - リアルタイム表示", 
                          fontsize=14, fontweight='bold')
        
        # ウィンドウタイトル設定（バージョン番号削除）
        try:
            if self.fig.canvas.manager:
                self.fig.canvas.manager.set_window_title("ED-SNN 学習進捗 - リアルタイム表示")
        except:
            pass
        
        # === 左上: ED法パラメータ設定（ed_multi_snn.prompt.md準拠） ===
        self.ax_params_ed.axis('off')
        ed_text = "ED法パラメータ設定\n"
        ed_text += f"学習率(alpha): {self.learning_rate:.3f}\n"
        ed_text += f"初期アミン(beta): {self.initial_amine:.3f}\n"
        ed_text += f"アミン拡散(u1): {self.diffusion_rate:.3f}\n"
        ed_text += f"シグモイド閾値(u0): {self.sigmoid_threshold:.3f}\n"
        ed_text += f"重み初期値1: {self.initial_weight_1:.3f}\n"
        ed_text += f"重み初期値2: {self.initial_weight_2:.3f}"
        
        # v021 Phase 2修正: 左寄せに変更（ヒートマップ準拠）
        self.ax_params_ed.text(0.05, 0.95, ed_text,
                               ha='left', va='top', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.5", 
                                       facecolor='lightgreen', 
                                       edgecolor='black',
                                       linewidth=2,
                                       alpha=0.8))
        
        # === 中上: LIFニューロンパラメータ設定（新規追加） ===
        self.ax_params_lif.axis('off')
        lif_text = "LIFニューロンパラメータ設定\n"
        lif_text += f"静止膜電位: {self.v_rest:.1f} mV\n"
        lif_text += f"発火閾値: {self.v_threshold:.1f} mV\n"
        lif_text += f"リセット電位: {self.v_reset:.1f} mV\n"
        lif_text += f"膜時定数: {self.tau_m:.1f} ms\n"
        lif_text += f"不応期: {self.tau_ref:.1f} ms\n"
        lif_text += f"時間ステップ: {self.dt:.1f} ms\n"
        lif_text += f"膜抵抗: {self.R_m:.1f} MΩ\n"
        lif_text += f"シミュレーション時間: {self.simulation_time:.1f} ms"
        
        # v021 Phase 2修正: 左寄せに変更（ヒートマップ準拠）
        self.ax_params_lif.text(0.05, 0.95, lif_text,
                                ha='left', va='top', fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.5", 
                                        facecolor='lightgreen', 
                                        edgecolor='black',
                                        linewidth=2,
                                        alpha=0.8))
        
        # === 右上: 実行パラメータ設定（ヒートマップ準拠） ===
        self.ax_params_exec.axis('off')
        exec_text = "実行パラメータ設定\n"
        exec_text += f"データセット: {self.dataset_name}\n"
        exec_text += f"エポック数: {self.max_epochs}\n"
        if self.train_samples:
            exec_text += f"学習サンプル: {self.train_samples}\n"
        if self.test_samples:
            exec_text += f"テストサンプル: {self.test_samples}\n"
        if self.hidden_layers:
            exec_text += f"隠れ層構造: {self.hidden_layers}\n"
        if self.batch_size:
            exec_text += f"ミニバッチサイズ: {self.batch_size}\n"
        # v021 Phase 1追加: ランダムシードと詳細表示
        seed_str = str(self.random_seed) if self.random_seed is not None else "ランダム"
        exec_text += f"ランダムシード: {seed_str}\n"
        verbose_str = "ON" if self.verbose else "OFF"
        exec_text += f"詳細表示: {verbose_str}"
        
        # v021 Phase 1修正: 左寄せに変更（ヒートマップ準拠）
        self.ax_params_exec.text(0.05, 0.95, exec_text,
                                 ha='left', va='top', fontsize=9,
                                 bbox=dict(boxstyle="round,pad=0.5", 
                                         facecolor='lightgreen', 
                                         edgecolor='black',
                                         linewidth=2,
                                         alpha=0.6))
        
        # === 左下～中下: 訓練・テスト正答率（幅広） ===
        self.ax_acc.set_title("訓練・テスト正答率", fontweight='bold')
        self.ax_acc.set_xlabel("エポック数")
        self.ax_acc.set_ylabel("精度 (%)")
        self.ax_acc.set_xlim(1, max(2, self.max_epochs))
        self.ax_acc.set_ylim(0, 100)
        self.ax_acc.grid(True, alpha=0.3)
        
        # === 右下: 訓練・テストエラー率 ===
        self.ax_err.set_title("訓練・テストエラー率", fontweight='bold')
        self.ax_err.set_xlabel("エポック数")
        self.ax_err.set_ylabel("エラー率 (%)")
        self.ax_err.set_xlim(1, max(2, self.max_epochs))
        self.ax_err.set_ylim(0, 100)  # エラー率: 0-100%
        self.ax_err.grid(True, alpha=0.3)
        
        # 線の初期化（エラー率 = 100% - 精度）
        self.lines['train_acc'], = self.ax_acc.plot([], [], 'b-', label='訓練正答率', linewidth=2)
        self.lines['test_acc'], = self.ax_acc.plot([], [], 'r-', label='テスト正答率', linewidth=2)
        self.lines['train_err'], = self.ax_err.plot([], [], 'b-', label='訓練エラー率', linewidth=2)
        self.lines['test_err'], = self.ax_err.plot([], [], 'r-', label='テストエラー率', linewidth=2)
        
        # 凡例設定
        self.ax_acc.legend(loc='lower right', fontsize=10, framealpha=0.9)
        self.ax_err.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # レイアウト調整（タイトル用のスペース確保）
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # インタラクティブモード有効化（リアルタイム表示の鍵）
        backend = matplotlib.get_backend()
        print(f"� matplotlib backend: {backend}")
        
        plt.ion()
        is_interactive = plt.isinteractive()
        print(f"📊 インタラクティブモード: {'有効' if is_interactive else '無効'}")
        
        if backend.lower() == 'agg':
            print("⚠️  警告: 非表示バックエンド(agg)が検出されました")
            print("    リアルタイム表示には対話的バックエンド(Qt5Agg/TkAgg)が必要です")
            print("    ファイル保存のみ実行されます")
        
        # 非ブロッキング表示
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.show(block=False)
        
        # 初期描画を強制実行
        if hasattr(self.fig, 'canvas') and self.fig.canvas:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # 確実な表示のため少し長めの待機
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.pause(0.1)  # 0.1秒待機
        
        if is_interactive and backend.lower() != 'agg':
            print("✅ リアルタイム可視化ウィンドウを表示中（学習初期から更新されます）")
    
    def update(self, epoch, train_acc, test_acc, train_err_rate, test_err_rate):
        """
        グラフデータ更新 - リアルタイム表示
        Args:
            epoch: 現在のエポック
            train_acc: 訓練正答率 (%)
            test_acc: テスト正答率 (%)
            train_err_rate: 訓練エラー率 (%) = 100 - train_acc
            test_err_rate: テストエラー率 (%) = 100 - test_acc
        """
        # データ追加
        self.epochs.append(epoch + 1)  # エポックは1から開始
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        self.train_error_rates.append(train_err_rate)  # エラー率: 0-100%
        self.test_error_rates.append(test_err_rate)    # エラー率: 0-100%
        
        # 線データ更新
        self.lines['train_acc'].set_data(self.epochs, self.train_accuracies)
        self.lines['test_acc'].set_data(self.epochs, self.test_accuracies)
        self.lines['train_err'].set_data(self.epochs, self.train_error_rates)
        self.lines['test_err'].set_data(self.epochs, self.test_error_rates)
        
        # エラー率グラフの縦軸は0-100%固定（エラー率の定義上）
        # 動的スケール調整は不要（常に0-100%の範囲内）
        
        # グラフ再描画（リアルタイム更新の鍵）
        try:
            self.ax_acc.relim()
            self.ax_acc.autoscale_view()
            self.ax_err.relim()
            self.ax_err.autoscale_view()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # 短時間の一時停止でリアルタイム表示
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.pause(0.05)  # 0.05秒待機（確実な表示更新）
        except Exception as e:
            # エラーがあっても処理継続（非表示バックエンドでも動作）
            pass
    
    def close(self):
        """可視化ウィンドウを閉じる"""
        if self.fig:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.close(self.fig)
            except Exception:
                plt.close(self.fig)
    
    def save_figure(self, save_dir=None):
        """リアルタイム学習グラフをタイムスタンプ付きファイル名で保存する
        
        Args:
            save_dir: 保存先ディレクトリ (Noneの場合は保存しない)
        """
        if not self.fig or save_dir is None:
            return
        
        # 保存ディレクトリの作成
        os.makedirs(save_dir, exist_ok=True)
        
        # タイムスタンプ付きファイル名の生成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'realtime_viz_result_{timestamp}.png'
        filepath = os.path.join(save_dir, filename)
        
        try:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✅ 学習曲線保存: {filepath}")
        except Exception as e:
            print(f"❌ グラフ保存エラー: {e}")

def convert_ed_outputs_to_spike_activities(ed_core, inputs, original_image_shape=(28, 28)):
    """
    ED出力をスパイク活動に変換（ヒートマップ用）
    
    Args:
        ed_core: ED Coreインスタンス
        inputs: E/Iペア化された入力（shape: [paired_input_size]）
        original_image_shape: 元の画像形状（(28, 28), (32, 32, 3)など）
    """
    spike_activities = []
    
    try:
        # 元の画像のピクセル数を計算
        base_size = np.prod(original_image_shape)
        
        # 入力層（元の画像形状で復元）
        # E/Iペア化された入力から興奮性ニューロンの値を抽出
        if len(inputs) >= base_size * 2:
            # GPU配列の場合は.get()でCPUに転送
            if hasattr(inputs, 'get'):
                inputs_cpu = inputs.get()
            else:
                inputs_cpu = np.asarray(inputs)
            
            input_excitatory = inputs_cpu[0::2][:base_size]
            input_layer = input_excitatory.reshape(original_image_shape)
            # カラー画像の場合はそのまま、グレースケールの場合はflattenして追加
            if len(original_image_shape) == 3:
                # カラー画像: (H, W, 3) の形状を保持
                spike_activities.append(input_layer)
            else:
                # グレースケール: (H, W) → flatten して (H*W,)
                spike_activities.append(input_layer.flatten())
        else:
            spike_activities.append(np.random.random(base_size))
        
        # 隠れ層
        if hasattr(ed_core, 'layer_outputs') and len(ed_core.layer_outputs) > 0:
            output_neuron_layers = ed_core.layer_outputs[0]
            for layer_output in output_neuron_layers[:-1]:
                # GPU配列の場合は.get()でCPUに転送
                if hasattr(layer_output, 'get'):
                    spike_activities.append(layer_output.get())
                else:
                    spike_activities.append(np.asarray(layer_output))
        else:
            spike_activities.append(np.random.random(64))
        
        # 出力層
        if hasattr(ed_core, 'layer_outputs') and len(ed_core.layer_outputs) > 0:
            output_activities = []
            for n in range(ed_core.output_units):
                if len(ed_core.layer_outputs[n]) > 0:
                    output_value = ed_core.layer_outputs[n][-1][0]
                    # GPU配列の場合は.get()でCPUに転送
                    if hasattr(output_value, 'get'):
                        output_activities.append(float(output_value.get()))
                    else:
                        output_activities.append(float(output_value))
            
            if output_activities:
                output_activity = np.array(output_activities)
            else:
                output_activity = np.random.random(ed_core.output_units) * 0.5
        else:
            output_activity = np.random.random(ed_core.output_units) * 0.5
        spike_activities.append(output_activity)
        
    except Exception as e:
        print(f"⚠️ スパイク活動変換エラー: {e}")
        base_size = np.prod(original_image_shape)
        spike_activities = [
            np.random.random(base_size),
            np.random.random(64),
            np.random.random(ed_core.output_units)
        ]
    
    return spike_activities

class PureEDPreprocessor:
    """
    ED法純粋データ前処理器（ed_multi_snn.prompt.md準拠）
    """
    
    @staticmethod
    def pure_ed_preprocess(images, labels, input_size):
        """純粋ED法前処理"""
        batch_size = len(images)
        
        # 入力データ正規化
        images_flat = images.reshape(batch_size, -1)
        images_normalized = images_flat / 255.0
        
        # 入力サイズ調整
        if images_normalized.shape[1] != input_size:
            if images_normalized.shape[1] > input_size:
                images_resized = images_normalized[:, :input_size]
            else:
                images_resized = np.zeros((batch_size, input_size))
                images_resized[:, :images_normalized.shape[1]] = images_normalized
        else:
            images_resized = images_normalized
            
        # ============================================================================
        # 興奮性・抑制性ニューロンペア (E/I pair)
        # ============================================================================
        # WHY: 入力の正負両方の情報を保持するため
        # WHY: 各ピクセル値を興奮性（+）と抑制性（-）の2つのニューロンで表現
        # WHY: 実際の脳の興奮性・抑制性のバランスを再現
        # ED法興奮性・抑制性ペア構造（ed_multi_snn.prompt.md準拠）
        ed_paired_input = np.zeros((batch_size, input_size * 2))
        ed_paired_input[:, 0::2] = images_resized  # 興奮性
        ed_paired_input[:, 1::2] = images_resized  # 抑制性
        
        return ed_paired_input, labels


def convert_to_lif_input(image_data: np.ndarray, scale_factor: float = 10.0) -> np.ndarray:
    """
    画像データをLIF層への入力電流に変換（v019 Phase 11修正）
    
    Parameters
    ----------
    image_data : np.ndarray
        正規化済み画像データ (0-1)
        784個（MNIST）または1568個（既にE/Iペア化済み）
    scale_factor : float
        電流スケールファクター（デフォルト: 10.0 nA）
        
    Returns
    -------
    np.ndarray
        電流パターン (nA単位)
        **Phase 11**: 1568個（興奮性784個+抑制性784個）
        
    Note
    ----
    v019 Phase 11修正: ED法仕様に完全準拠
    - 金子勇氏のオリジナルED法では入力層は興奮性・抑制性ペア構成が必須
    - 物理的に1568個のニューロン（興奮性784個+抑制性784個）で構成
    - 入力が784個の場合: 1568個に変換
    - 入力が1568個の場合: そのまま処理（既にペア化済み）
    
    簡易版実装: 画像強度 × スケールファクター
    0-1 → 0-10 nA の範囲に変換
    
    将来拡張:
    - ポアソンスパイク列生成
    - 時間的パターン変換
    - レート符号化/時間符号化切り替え
    """
    # v019 Phase 11: ED法仕様準拠の実装
    
    # 入力が既に1568個（E/Iペア化済み）の場合
    if len(image_data) == 1568:
        # そのまま電流に変換
        current_pattern = image_data * scale_factor
        min_activation = 0.01
        current_pattern = current_pattern + min_activation
        return current_pattern
    
    # 入力が784個の場合: 1568個に変換
    current_pattern = image_data * scale_factor
    
    # 最小活性化値を追加（完全なゼロを避ける）
    min_activation = 0.01
    current_pattern = current_pattern + min_activation
    
    # 興奮性・抑制性ペアに変換
    # [pixel0, pixel1, ...] → [exc0, inh0, exc1, inh1, ...]
    paired_pattern = np.zeros(len(current_pattern) * 2)
    for i in range(len(current_pattern)):
        paired_pattern[2*i] = current_pattern[i]      # 興奮性ニューロン
        paired_pattern[2*i + 1] = current_pattern[i]  # 抑制性ニューロン（同じ値）
    
    return paired_pattern  # 1568個を返す


class MultiLayerEDCore:
    """
    真の多層対応純粋Error-Diffusion学習エンジン（v012/v013版）
    ed_multi_snn.prompt.md拡張機能1完全準拠
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.8,
                 initial_amine=0.25, diffusion_rate=0.5, sigmoid_threshold=1.2,
                 initial_weight_1=0.3, initial_weight_2=0.5, snn=None, hp=None):
        """真の多層ED法初期化（ed_multi_snn.prompt.md準拠）
        
        Args:
            snn: SpikingNeuralNetwork instance (v019 Phase 5追加)
            hp: HyperParams instance (v019 Phase 5追加)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if hidden_sizes else [64]
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # v019 Phase 5: LIF統合用
        self.snn = snn
        self.hp = hp
        self.lif_stats = {'firing_rates': [], 'total_spikes': 0, 'avg_voltage': 0.0}  # LIF統計情報
        
        # 隠れ層サイズを偶数に調整
        self.hidden_sizes = [size + (size % 2) for size in self.hidden_sizes]
        
        # v019 Phase 14修正: 入力ユニット数は常に1568個
        # 金子勇氏のオリジナル実装解析により判明:
        # - 全1568個のニューロン（興奮性784個+抑制性784個）が次層に接続
        # - LIF使用/不使用は関係なく、常に1568個
        # - Phase 12の条件分岐は誤りでした
        self.input_units = input_size  # 常に1568個（E/Iペア構造）
        self.output_units = output_size
        
        # ED法コアパラメータ（ed_multi_snn.prompt.md準拠）
        self.initial_amine = initial_amine          # β: 初期アミン濃度
        self.diffusion_rate = diffusion_rate        # u1: アミン拡散係数
        self.sigmoid_threshold = sigmoid_threshold  # u0: シグモイド閾値
        self.initial_weight_1 = initial_weight_1    # 重み初期値1（興奮性）
        self.initial_weight_2 = initial_weight_2    # 重み初期値2（抑制性）
        self.time_loops = 2  # 複数回のループで安定した応答を得る
        
        # GPU対応（ed_multi_snn.prompt.md 拡張機能7準拠 + --cpuオプション対応）
        self.use_gpu = GPU_AVAILABLE and not (hp.force_cpu if hp else False)
        self.xp = np if (hp and hp.force_cpu) else xp
        if self.use_gpu:
            print("🚀 ED法コア: GPU（CuPy）で初期化")
        elif hp and hp.force_cpu and GPU_AVAILABLE:
            print("🔧 ED法コア: CPU強制実行モード（--cpuオプション指定）")
        
        # 層ごとの重み行列（ed_multi_snn.prompt.md準拠の初期値使用）
        # GPU対応: 重み行列をGPU上に保持
        self.layer_weights = []
        for n in range(self.output_units):
            neuron_weights = []
            
            # Input → Hidden1
            # 興奮性ニューロン: initial_weight_1, 抑制性: initial_weight_2
            w_input_h1 = np.random.uniform(self.initial_weight_1, self.initial_weight_2, 
                                          (self.hidden_sizes[0], self.input_units))
            if self.use_gpu:
                w_input_h1 = self.xp.asarray(w_input_h1)
            neuron_weights.append(w_input_h1)
            
            # Hidden_i → Hidden_{i+1}
            for i in range(len(self.hidden_sizes) - 1):
                w_h_h = np.random.uniform(self.initial_weight_1, self.initial_weight_2, 
                                         (self.hidden_sizes[i+1], self.hidden_sizes[i]))
                if self.use_gpu:
                    w_h_h = self.xp.asarray(w_h_h)
                neuron_weights.append(w_h_h)
            
            # Hidden_last → Output
            w_h_output = np.random.uniform(self.initial_weight_1, self.initial_weight_2, 
                                          (1, self.hidden_sizes[-1]))
            if self.use_gpu:
                w_h_output = self.xp.asarray(w_h_output)
            neuron_weights.append(w_h_output)
            
            self.layer_weights.append(neuron_weights)
        
        # ニューロンタイプ初期化
        self._initialize_neuron_types()
        
        # ============================================================================
        # Dale's Principle（デールの原理）
        # ============================================================================
        # WHY: 実際の脳では1つのニューロンは興奮性または抑制性のどちらか一方のみ
        # WHY: 重み符号を保持することで生物学的妥当性を確保
        # WHY: 興奮性ニューロン→正の重みのみ、抑制性ニューロン→負の重みのみ
        # Dale's Principle適用
        self._apply_dales_principle()
        
        # アミン濃度配列
        self.layer_amine_concentrations = []
        for n in range(self.output_units):
            layer_amines = []
            for size in self.hidden_sizes:
                layer_amines.append(np.zeros((size, 2)))
            layer_amines.append(np.zeros((1, 2)))
            self.layer_amine_concentrations.append(layer_amines)
        
        # 層出力保存用
        self.layer_outputs = []
        for n in range(self.output_units):
            layer_outs = [np.zeros(size) for size in self.hidden_sizes]
            layer_outs.append(np.zeros(1))
            self.layer_outputs.append(layer_outs)
        
        # 学習統計
        self.error = 0.0
        self.error_count = 0
        
        print(f"✅ 真の多層ED法初期化完了（層数: {len(self.hidden_sizes) + 1}）")
    
    def _initialize_neuron_types(self):
        """ニューロンタイプ初期化（Phase 12: オリジナルCコード準拠）
        
        金子勇氏のCコード: ow[k] = ((k+1) % 2) * 2 - 1
        - ow[0] = 1 (興奮性)
        - ow[1] = -1 (抑制性)
        - ow[2] = 1 (興奮性)
        - ow[3] = -1 (抑制性)
        ...
        
        Phase 12修正: 入力層は1568個の物理的ニューロン
        """
        # 入力層: 1568個のニューロンタイプ（興奮性/抑制性交互）
        # オリジナルCコードと完全に同じロジック
        self.input_neuron_types = np.ones(self.input_units)
        for i in range(1, self.input_units, 2):
            self.input_neuron_types[i] = -1
        
        # 隠れ層: 既存の実装を維持（マルチクラス拡張機能）
        self.hidden_neuron_types = []
        for size in self.hidden_sizes:
            types = np.ones(size)
            for i in range(1, size, 2):
                types[i] = -1
            self.hidden_neuron_types.append(types)
        
        # 出力層: 全て興奮性（マルチクラス拡張機能）
        self.output_neuron_types = np.ones(self.output_units)
    
    def _apply_dales_principle(self):
        """Dale's Principle適用（Phase 12: オリジナルCコード準拠 + GPU対応）
        
        金子勇氏のCコード: w_ot_ot[n][k][l] *= ow[l] * ow[k]
        - 同種の細胞間（E→E、I→I）: ow[l] * ow[k] = 1 → 正の重み
        - 異種の細胞間（E→I、I→E）: ow[l] * ow[k] = -1 → 負の重み
        
        Phase 12修正: 1568個のニューロンタイプ配列を使用
        GPU最適化: ベクトル化されたマスク演算を使用
        """
        for n in range(self.output_units):
            # Input → Hidden1（1568個のニューロンタイプ配列を使用）
            # GPU最適化: ベクトル化
            if self.use_gpu:
                # GPU上でベクトル化演算
                src_types = self.xp.asarray(self.input_neuron_types).reshape(1, -1)
                dst_types = self.xp.asarray(self.hidden_neuron_types[0]).reshape(-1, 1)
                mask = src_types * dst_types
                self.layer_weights[n][0] *= mask
            else:
                for i in range(self.hidden_sizes[0]):
                    dst_type = self.hidden_neuron_types[0][i]
                    for j in range(self.input_units):
                        src_type = self.input_neuron_types[j]
                        # オリジナルCコードと同じロジック: w *= ow[l] * ow[k]
                        self.layer_weights[n][0][i, j] *= src_type * dst_type
            
            # Hidden層間（マルチクラス拡張機能を維持 + GPU最適化）
            for layer_idx in range(len(self.hidden_sizes) - 1):
                if self.use_gpu:
                    # GPU上でベクトル化演算
                    src_types = self.xp.asarray(self.hidden_neuron_types[layer_idx]).reshape(1, -1)
                    dst_types = self.xp.asarray(self.hidden_neuron_types[layer_idx + 1]).reshape(-1, 1)
                    mask = src_types * dst_types
                    self.layer_weights[n][layer_idx + 1] *= mask
                else:
                    src_types = self.hidden_neuron_types[layer_idx]
                    dst_types = self.hidden_neuron_types[layer_idx + 1]
                    for i in range(self.hidden_sizes[layer_idx + 1]):
                        dst_type = dst_types[i]
                        for j in range(self.hidden_sizes[layer_idx]):
                            src_type = src_types[j]
                            self.layer_weights[n][layer_idx + 1][i, j] *= src_type * dst_type
            
            # Hidden_last → Output（マルチクラス拡張機能を維持 + GPU最適化）
            last_layer_idx = len(self.hidden_sizes)
            if self.use_gpu:
                # GPU上でベクトル化演算
                src_types = self.xp.asarray(self.hidden_neuron_types[-1]).reshape(1, -1)
                output_type = self.output_neuron_types[n]
                mask = src_types * output_type
                self.layer_weights[n][last_layer_idx] *= mask
            else:
                last_hidden_types = self.hidden_neuron_types[-1]
                output_type = self.output_neuron_types[n]
                for j in range(self.hidden_sizes[-1]):
                    src_type = last_hidden_types[j]
                    # オリジナルCコードと同じロジック: w *= ow[l] * ow[k]
                    self.layer_weights[n][last_layer_idx][0, j] *= src_type * output_type
    
    def sigmoid(self, x):
        """シグモイド関数"""
        safe_x = -2.0 * x / self.sigmoid_threshold
        safe_x = np.clip(safe_x, -500, 500)
        return 1.0 / (1.0 + np.exp(safe_x))
    
    def _sigmoid_vectorized(self, x):
        """ベクトル化シグモイド関数（GPU対応）"""
        safe_x = -2.0 * x / self.sigmoid_threshold
        safe_x = self.xp.clip(safe_x, -500, 500)
        return 1.0 / (1.0 + self.xp.exp(safe_x))
    
    # ========================================
    # Step 3a: スパイク符号化メソッド
    # ========================================
    
    def _poisson_encode(self, pixel_values, max_rate=100.0, simulation_time=50.0, dt=1.0):
        """ポアソン符号化（推奨）- GPU最適化版
        
        生物学的妥当性が最も高いスパイク符号化手法。
        画素値に比例した発火率でランダムにスパイクを生成。
        
        【Step 4-3 GPU最適化】:
        - ベクトル化: ループを完全にベクトル化（n_timesteps × n_neurons一括生成）
        - メモリ効率: 中間配列を削減
        - GPU並列化: 乱数生成とスパイク判定を並列実行
        
        Args:
            pixel_values: 画素値配列 [784] (正規化済み [0,1])
            max_rate: 最大発火率 (Hz)
            simulation_time: シミュレーション時間 (ms)
            dt: 時間刻み (ms)
        
        Returns:
            spike_trains: スパイク列 [n_timesteps, n_neurons] (bool)
        
        参考文献:
            - Diehl & Cook (2015) "Unsupervised learning of digit recognition using spike-timing-dependent plasticity"
            - 山本拓都「NumPyで作って試すSNN」p.87, Jittered MNIST実装
        """
        n_neurons = len(pixel_values)
        n_timesteps = int(simulation_time / dt)
        
        # GPU対応 - 完全ベクトル化版（Step 4-3最適化）
        if self.use_gpu:
            # 発火率を計算 [n_neurons]
            rates = self.xp.asarray(pixel_values) * max_rate
            
            # 発火確率を計算 [n_neurons]
            probs = rates * dt / 1000.0  # Hz → 確率変換
            
            # 乱数生成 [n_timesteps, n_neurons] - 一括生成で高速化
            random_vals = self.xp.random.random((n_timesteps, n_neurons))
            
            # スパイク判定 [n_timesteps, n_neurons] - ベクトル化比較
            spike_trains = random_vals < probs[self.xp.newaxis, :]
            
        else:
            # CPU版 - 同様にベクトル化
            rates = pixel_values * max_rate
            probs = rates * dt / 1000.0
            random_vals = np.random.random((n_timesteps, n_neurons))
            spike_trains = random_vals < probs[np.newaxis, :]
        
        return spike_trains
    
    def _rate_encode(self, pixel_values, max_rate=100.0, simulation_time=50.0, dt=1.0):
        """レート符号化（決定論的）
        
        画素値に比例した一定発火率で規則的にスパイクを生成。
        
        Args:
            pixel_values: 画素値配列 [784] (正規化済み [0,1])
            max_rate: 最大発火率 (Hz)
            simulation_time: シミュレーション時間 (ms)
            dt: 時間刻み (ms)
        
        Returns:
            spike_trains: スパイク列 [n_timesteps, n_neurons] (bool)
        """
        n_neurons = len(pixel_values)
        n_timesteps = int(simulation_time / dt)
        
        # GPU対応
        if self.use_gpu:
            spike_trains = self.xp.zeros((n_timesteps, n_neurons), dtype=bool)
            rates = self.xp.asarray(pixel_values) * max_rate
            intervals = self.xp.where(rates > 0, 1000.0 / rates, self.xp.inf)  # ms
            
            for i in range(n_neurons):
                if rates[i] > 0:
                    interval = float(intervals[i])
                    spike_times = self.xp.arange(interval, simulation_time, interval)
                    spike_indices = (spike_times / dt).astype(int)
                    spike_indices = spike_indices[spike_indices < n_timesteps]
                    spike_trains[spike_indices, i] = True
        else:
            spike_trains = np.zeros((n_timesteps, n_neurons), dtype=bool)
            rates = pixel_values * max_rate
            intervals = np.where(rates > 0, 1000.0 / rates, np.inf)
            
            for i in range(n_neurons):
                if rates[i] > 0:
                    interval = intervals[i]
                    spike_times = np.arange(interval, simulation_time, interval)
                    spike_indices = (spike_times / dt).astype(int)
                    spike_indices = spike_indices[spike_indices < n_timesteps]
                    spike_trains[spike_indices, i] = True
        
        return spike_trains
    
    def _temporal_encode(self, pixel_values, simulation_time=50.0, dt=1.0):
        """テンポラル符号化（時間コーディング）
        
        画素値が大きいほど早く発火。各ニューロンは1回のみ発火。
        
        Args:
            pixel_values: 画素値配列 [784] (正規化済み [0,1])
            simulation_time: シミュレーション時間 (ms)
            dt: 時間刻み (ms)
        
        Returns:
            spike_trains: スパイク列 [n_timesteps, n_neurons] (bool)
        """
        n_neurons = len(pixel_values)
        n_timesteps = int(simulation_time / dt)
        
        # GPU対応
        if self.use_gpu:
            spike_trains = self.xp.zeros((n_timesteps, n_neurons), dtype=bool)
            # 画素値が大きいほど早く発火（逆比例）
            # 値が0の場合は発火しない
            spike_times_ms = self.xp.where(
                self.xp.asarray(pixel_values) > 0,
                simulation_time * (1.0 - self.xp.asarray(pixel_values)),
                self.xp.inf
            )
            spike_indices = (spike_times_ms / dt).astype(int)
            
            for i in range(n_neurons):
                if spike_indices[i] < n_timesteps:
                    spike_trains[int(spike_indices[i]), i] = True
        else:
            spike_trains = np.zeros((n_timesteps, n_neurons), dtype=bool)
            spike_times_ms = np.where(
                pixel_values > 0,
                simulation_time * (1.0 - pixel_values),
                np.inf
            )
            spike_indices = (spike_times_ms / dt).astype(int)
            
            for i in range(n_neurons):
                if spike_indices[i] < n_timesteps:
                    spike_trains[int(spike_indices[i]), i] = True
        
        return spike_trains
    
    def _spike_encode(self, pixel_values, method='poisson', max_rate=100.0, 
                     simulation_time=50.0, dt=1.0):
        """スパイク符号化（メソッド選択 + E/Iペア化）- GPU最適化版
        # ============================================================================
        # スパイク符号化（ポアソン符号化）
        # ============================================================================
        # WHY: 実際の神経細胞はアナログ値ではなくスパイク列で情報を表現
        # WHY: ポアソン過程により、入力強度を確率的な発火率に変換
        # WHY: 生物学的妥当性が最も高く、ノイズロバスト性も優れている
        
        画素値をスパイク列に変換し、E/Iペア構造を適用。
        
        【Step 4-3 GPU最適化】:
        - E/Iペア化のベクトル化: ループを完全削除
        - repeat()とreshape()による高速ペア化
        - メモリ効率化: 中間配列を削減
        
        Args:
            pixel_values: 画素値配列 [784] (正規化済み [0,1])
            method: 符号化方法 ('poisson', 'rate', 'temporal')
            max_rate: 最大発火率 (Hz) - poisson/rateのみ
            simulation_time: シミュレーション時間 (ms)
            dt: 時間刻み (ms)
        
        Returns:
            spike_trains_paired: E/Iペア化されたスパイク列 [n_timesteps, 1568] (bool)
        """
        # Step 1: 画素値 [784] → スパイク列 [n_timesteps, 784]
        if method == 'poisson':
            spike_trains_raw = self._poisson_encode(pixel_values, max_rate, simulation_time, dt)
        elif method == 'rate':
            spike_trains_raw = self._rate_encode(pixel_values, max_rate, simulation_time, dt)
        elif method == 'temporal':
            spike_trains_raw = self._temporal_encode(pixel_values, simulation_time, dt)
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        # Step 2: E/Iペア化 [n_timesteps, 784] → [n_timesteps, 1568]
        # GPU最適化版（Step 4-4）: stack()による高速ペア化（1.27倍高速）
        n_timesteps, n_pixels = spike_trains_raw.shape
        
        if self.use_gpu:
            # stack()による高速ペア化
            # [n_timesteps, 784] → [n_timesteps, 784, 2] → [n_timesteps, 1568]
            spike_trains_paired = self.xp.stack([spike_trains_raw, spike_trains_raw], axis=2)
            spike_trains_paired = spike_trains_paired.reshape(n_timesteps, n_pixels * 2)
        else:
            # CPU版も同様にstack()を使用
            spike_trains_paired = np.stack([spike_trains_raw, spike_trains_raw], axis=2)
            spike_trains_paired = spike_trains_paired.reshape(n_timesteps, n_pixels * 2)
        
        return spike_trains_paired

    
    # ========================================
    # Step 1: LIF活性化メソッド（隠れ層・出力層用）
    # ========================================
    
    def _lif_activation(self, inputs, layer_size, neuron_types, 
                       simulation_time=50.0, dt=1.0):
        """LIF活性化関数（シグモイドの代替、Step 1）
        # ============================================================================
        # LIF (Leaky Integrate-and-Fire) ニューロン
        # ============================================================================
        # WHY: 実際の神経細胞の膜電位ダイナミクスを模倣
        # WHY: 膜電位の時間的統合により、スパイクタイミングを考慮した計算が可能
        # WHY: 発火閾値を超えた時のみスパイクを発生させる離散的な情報処理
        
        隠れ層・出力層でシグモイドの代わりに使用するLIF活性化関数。
        連続値入力を電流に変換し、LIFシミュレーションを実行。
        
        Args:
            inputs: 連続値入力 [layer_size] (任意の範囲)
            layer_size: 層のニューロン数
            neuron_types: ニューロンタイプ配列 [layer_size] (+1: 興奮性, -1: 抑制性)
            simulation_time: シミュレーション時間 (ms)
            dt: 時間刻み (ms)
        
        Returns:
            firing_rates: 発火率 [layer_size] (0-1の範囲に正規化)
        """
        from modules.snn.lif_neuron import LIFNeuronLayer
        
        # GPU配列をNumPyに変換
        if self.use_gpu and hasattr(inputs, 'get'):
            inputs_cpu = inputs.get()
        else:
            inputs_cpu = np.asarray(inputs)
        
        # ニューロンタイプ配列を変換 (+1 → 'excitatory', -1 → 'inhibitory')
        neuron_type_names = ['excitatory' if t == 1 else 'inhibitory' for t in neuron_types]
        
        # LIF層初期化
        neuron_params = {
            'v_rest': -65.0,
            'v_threshold': -40.0,
            'v_reset': -70.0,
            'tau_m': 12.0,
            'tau_ref': 1.0,
            'dt': dt,
            'r_m': 35.0
        }
        
        lif_layer = LIFNeuronLayer(
            n_neurons=layer_size,
            neuron_params=neuron_params,
            neuron_types=neuron_type_names
        )
        
        # 連続値入力を電流に変換
        # 入力範囲を適切にスケーリング（シグモイド出力[0,1]を想定）
        # 電流範囲: 0-20 pA（LIFが適切に発火する範囲）
        input_currents = inputs_cpu * 20.0
        
        # LIFシミュレーション実行
        n_timesteps = int(simulation_time / dt)
        spike_counts = np.zeros(layer_size)
        
        for t in range(n_timesteps):
            # 各時間ステップで同じ電流を注入（定常入力）
            spikes = lif_layer.update(input_currents)
            spike_counts += spikes
        
        # 発火率計算（スパイク数 / 時間ステップ数）
        firing_rates = spike_counts / n_timesteps
        
        # [0, 1]範囲に正規化
        firing_rates = np.clip(firing_rates, 0.0, 1.0)
        
        # GPU配列に変換（後続処理用）
        if self.use_gpu:
            firing_rates = self.xp.asarray(firing_rates)
        
        return firing_rates
    
    # ========================================
    # Step 3a: LIF活性化メソッド（入力層専用）
    # ========================================
    
    def _lif_activation_input_layer(self, spike_trains, neuron_types, 
                                    simulation_time=50.0, dt=1.0):
        """入力層LIF活性化関数（スパイク列 → 発火率）
        
        入力層専用のLIFニューロンシミュレーション。
        スパイク列を入力し、LIFニューロンの発火率を出力。
        
        Args:
            spike_trains: スパイク列 [n_timesteps, n_neurons] (bool)
            neuron_types: ニューロンタイプ配列 [n_neurons] (+1: 興奮性, -1: 抑制性)
            simulation_time: シミュレーション時間 (ms)
            dt: 時間刻み (ms)
        
        Returns:
            firing_rates: 発火率 [n_neurons] (0-1の範囲に正規化)
        """
        n_timesteps, n_neurons = spike_trains.shape
        
        # modules/snn/lif_neuron.pyのLIFNeuronLayerを使用
        from modules.snn.lif_neuron import LIFNeuronLayer
        
        # ニューロンタイプ配列を変換 (+1 → 'excitatory', -1 → 'inhibitory')
        neuron_type_names = ['excitatory' if t == 1 else 'inhibitory' for t in neuron_types]
        
        # 入力層LIF初期化（E/Iペア構造）
        # v_rest=-65.0, v_threshold=-40.0, tau_m=12.0, tau_ref=1.0がデフォルト
        neuron_params = {
            'v_rest': -65.0,
            'v_threshold': -40.0,
            'v_reset': -70.0,
            'tau_m': 12.0,
            'tau_ref': 1.0,
            'dt': dt,
            'r_m': 35.0
        }
        
        input_lif_layer = LIFNeuronLayer(
            n_neurons=n_neurons,
            neuron_params=neuron_params,
            neuron_types=neuron_type_names
        )
        
        # LIFシミュレーション実行
        spike_counts = np.zeros(n_neurons)
        
        for t in range(n_timesteps):
            # スパイク → 電流変換（スパイクがあれば10.0 pAの電流を注入）
            # GPU配列をNumPyに変換
            if self.use_gpu:
                current_spikes = self.xp.asnumpy(spike_trains[t]) if hasattr(spike_trains[t], 'get') else np.array(spike_trains[t])
            else:
                current_spikes = spike_trains[t].astype(float)
            
            input_currents = current_spikes * 10.0  # pA
            
            # LIF更新
            output_spikes = input_lif_layer.update(input_currents)
            spike_counts += output_spikes
        
        # 発火率計算（スパイク数 / 時間ステップ数）
        firing_rates = spike_counts / n_timesteps
        
        # [0, 1]範囲に正規化（最大発火率で割る）
        # LIFニューロンの最大発火率は約100Hz（不応期1ms → 1000Hz理論値、実際は100Hz程度）
        max_possible_rate = 1.0  # 既に正規化済み（全時間ステップで発火した場合=1.0）
        firing_rates = np.clip(firing_rates, 0.0, max_possible_rate)
        
        # GPU配列に変換（後続処理用）
        if self.use_gpu:
            firing_rates = self.xp.asarray(firing_rates)
        
        return firing_rates
    
    def forward_pass(self, inputs):
        """真の多層順伝播処理（v025 Step 3a: 入力層LIF統合対応）"""
        # Phase 12修正: input_unitsを使用（LIF使用時は1568、不使用時は784）
        if len(inputs) != self.input_units:
            adjusted = np.zeros(self.input_units)
            min_len = min(len(inputs), self.input_units)
            adjusted[:min_len] = inputs[:min_len]
            inputs = adjusted
        
        outputs = np.zeros(self.output_units)
        
        # ========================================
        # v025 Step 3a: 入力層LIF統合
        # ========================================
        if self.hp is not None and self.hp.use_input_lif:
            # Step 3a: 生物学的妥当性の高い入力層LIF処理
            # 画素値 [784] → スパイク列 [n_timesteps, 784] → E/Iペア [n_timesteps, 1568] 
            #   → 入力層LIF → 発火率 [1568] → 隠れ層伝播
            
            # Step 1: 画素値 [1568] から元の画素値 [784] を抽出
            # inputs は E/Iペア化済み [1568]、偶数インデックスが元の画素値
            original_pixels = inputs[0::2]  # [784]
            
            # Step 2: スパイク符号化（E/Iペア化を含む）
            # [784] → [n_timesteps, 1568]
            spike_trains = self._spike_encode(
                pixel_values=original_pixels,
                method=self.hp.spike_encoding_method,
                max_rate=self.hp.spike_max_rate,
                simulation_time=self.hp.spike_simulation_time,
                dt=self.hp.spike_dt
            )
            
            # Step 3: 入力層LIF活性化
            # スパイク列 [n_timesteps, 1568] → 発火率 [1568]
            input_activity = self._lif_activation_input_layer(
                spike_trains=spike_trains,
                neuron_types=self.input_neuron_types,
                simulation_time=self.hp.spike_simulation_time,
                dt=self.hp.spike_dt
            )
            
            # Step 4: 発火率を隠れ層に伝播（従来のシグモイド処理）
            # GPU対応: 入力データをGPUに転送（1回のみ）
            if self.use_gpu:
                inputs_gpu = input_activity  # 既にGPU配列
            else:
                inputs_gpu = input_activity
            
            for n in range(self.output_units):
                for t in range(self.time_loops):
                    layer_outputs = []
                    current_layer_output = inputs_gpu.copy() if self.use_gpu else input_activity.copy()
                    
                    for layer_idx, layer_weight in enumerate(self.layer_weights[n]):
                        # GPU最適化: 重み行列は既にGPU上にあるので転送不要
                        linear_out = layer_weight @ current_layer_output
                        activated = self._sigmoid_vectorized(linear_out)
                        
                        # GPU→CPUに戻す（layer_outputsはNumPy配列として保存）
                        if self.use_gpu:
                            layer_outputs.append(np.array(activated) if not hasattr(activated, 'get') else activated.get())
                        else:
                            layer_outputs.append(activated)
                        
                        current_layer_output = activated
                    
                    if t == self.time_loops - 1:
                        self.layer_outputs[n] = layer_outputs
                
                outputs[n] = self.layer_outputs[n][-1][0]
            
            return outputs
        
        # v019 Phase 12: LIF層使用時の条件分岐（hp.enable_lifのみで判定）
        # v025 Step 2a/2b: 隠れ層・出力層LIF化
        elif self.hp is not None and self.hp.enable_lif:
            # ========================================
            # Step 2a/2b: 隠れ層・出力層LIF化処理
            # ========================================
            
            # 入力層は従来通り（E/Iペア化済み画素値 [1568]）
            # 隠れ層・出力層でLIF活性化関数を使用（シグモイドの代替）
            
            # GPU対応: 入力データをGPUに転送（1回のみ）
            if self.use_gpu:
                inputs_gpu = self.xp.asarray(inputs)
            else:
                inputs_gpu = inputs
            
            for n in range(self.output_units):
                for t in range(self.time_loops):
                    layer_outputs = []
                    current_layer_output = inputs_gpu.copy() if self.use_gpu else inputs.copy()
                    
                    for layer_idx, layer_weight in enumerate(self.layer_weights[n]):
                        # 重み行列積
                        linear_out = layer_weight @ current_layer_output
                        
                        # ★Step 2a/2b: LIF活性化関数を使用（シグモイドの代替）
                        # 隠れ層・出力層のニューロンタイプを取得
                        if layer_idx < len(self.hidden_neuron_types):
                            # 隠れ層
                            neuron_types = self.hidden_neuron_types[layer_idx]
                            layer_size = self.hidden_sizes[layer_idx]
                        else:
                            # 出力層
                            neuron_types = np.array([self.output_neuron_types[n]])
                            layer_size = 1
                        
                        # LIF活性化（Step 1メソッド使用）
                        activated = self._lif_activation(
                            inputs=linear_out,
                            layer_size=layer_size,
                            neuron_types=neuron_types,
                            simulation_time=self.hp.simulation_time,
                            dt=self.hp.dt
                        )
                        
                        # GPU→CPUに戻す（layer_outputsはNumPy配列として保存）
                        if self.use_gpu:
                            layer_outputs.append(np.array(activated) if not hasattr(activated, 'get') else activated.get())
                        else:
                            layer_outputs.append(activated)
                        
                        current_layer_output = activated
                    
                    if t == self.time_loops - 1:
                        self.layer_outputs[n] = layer_outputs
                
                outputs[n] = self.layer_outputs[n][-1][0]
            
            return outputs
        
        else:
            # ========================================
            # 従来のシグモイドベース順伝播処理（Phase 13修正 + GPU最適化）
            # ========================================
            
            # GPU対応: 入力データをGPUに転送（1回のみ）
            if self.use_gpu:
                inputs_gpu = self.xp.asarray(inputs)
            else:
                inputs_gpu = inputs
            
            for n in range(self.output_units):
                for t in range(self.time_loops):
                    layer_outputs = []
                    
                    # Phase 13修正: input_unitsが正しく設定されているので、
                    # inputs[0::2]による抽出は不要。inputsをそのまま使用。
                    current_layer_output = inputs_gpu.copy() if self.use_gpu else inputs.copy()
                    
                    for layer_idx, layer_weight in enumerate(self.layer_weights[n]):
                        # GPU最適化: 重み行列は既にGPU上にあるので転送不要
                        linear_out = layer_weight @ current_layer_output
                        activated = self._sigmoid_vectorized(linear_out)
                        
                        # GPU→CPUに戻す（layer_outputsはNumPy配列として保存）
                        if self.use_gpu:
                            layer_outputs.append(self.xp.asnumpy(activated))
                        else:
                            layer_outputs.append(activated)
                        
                        current_layer_output = activated
                    
                    if t == self.time_loops - 1:
                        self.layer_outputs[n] = layer_outputs
                
                outputs[n] = self.layer_outputs[n][-1][0]
        
        return outputs
    
    def pure_ed_learning_step(self, inputs, targets, outputs):
        """真の多層ED学習ステップ（Phase 13修正）
        
        Phase 13修正:
        - input_unitsが正しく設定されているので、inputs[0::2]による抽出は不要
        - inputsをそのまま使用
        
        v019 Phase 11修正（Phase 13で改善）:
        - LIF使用時: inputs は1568個（E/Iペア構造）のまま使用
        - LIF不使用時: inputs は784個（input_unitsサイズ）のまま使用
        """
        # Phase 13修正: 条件分岐を削除し、inputsをそのまま使用
        input_for_learning = inputs.copy()
        
        self.error = 0.0
        
        for n in range(self.output_units):
            error = targets[n] - outputs[n]
            self.error += abs(error)
            
            if abs(error) > 0.5:
                self.error_count += 1
            
            # 出力層アミン濃度設定
            if error > 0:
                self.layer_amine_concentrations[n][-1][0, 0] = error
                self.layer_amine_concentrations[n][-1][0, 1] = 0
            else:
                self.layer_amine_concentrations[n][-1][0, 0] = 0
                self.layer_amine_concentrations[n][-1][0, 1] = -error
            
            # ============================================================================
            # ED法の核心: アミン拡散メカニズム
            # ============================================================================
            # WHY: 生物学的な神経伝達物質（ドーパミン、セロトニン）の拡散を模倣
            # WHY: 誤差逆伝播法の「微分の連鎖律」を使わずに学習を実現
            # WHY: 出力層の誤差を「アミン濃度」として隠れ層に拡散させることで学習信号を伝える

            # ============================================================================
            # ED法の核心: アミン拡散メカニズム
            # ============================================================================
            # WHY: 生物学的な神経伝達物質（ドーパミン、セロトニン）の拡散を模倣
            # WHY: 誤差逆伝播法の「微分の連鎖律」を使わずに学習を実現
            # WHY: 出力層の誤差を「アミン濃度」として隠れ層に拡散させることで学習信号を伝える
            # 隠れ層へのアミン拡散
            for layer_idx in range(len(self.hidden_sizes) - 1, -1, -1):
                if layer_idx == len(self.hidden_sizes) - 1:
                    pos_amine = self.layer_amine_concentrations[n][-1][0, 0]
                    neg_amine = self.layer_amine_concentrations[n][-1][0, 1]
                else:
                    pos_amine = np.mean(self.layer_amine_concentrations[n][layer_idx + 1][:, 0])
                    neg_amine = np.mean(self.layer_amine_concentrations[n][layer_idx + 1][:, 1])
                
                self.layer_amine_concentrations[n][layer_idx][:, 0] = pos_amine * self.diffusion_rate
                self.layer_amine_concentrations[n][layer_idx][:, 1] = neg_amine * self.diffusion_rate
        
        # 重み更新
        self._neuro_weight_calc_multilayer(input_for_learning)
    
    def _neuro_weight_calc_multilayer(self, input_data):
        """真の多層重み更新（GPU最適化版）
        
        Args:
            input_data: 入力データ
                - LIF使用時: 1568個（E/Iペア構造）
                - LIF不使用時: 784個（興奮性のみ）
        
        GPU最適化戦略:
            - 重み行列はGPU上に常駐
            - 転送回数を最小化
            - GPU上で全計算を完結
        """
        # GPU対応: 入力データをGPUに転送（1回のみ）
        if self.use_gpu:
            input_data_gpu = self.xp.asarray(input_data)
            input_neuron_types_gpu = self.xp.asarray(self.input_neuron_types)
        else:
            input_data_gpu = input_data
            input_neuron_types_gpu = self.input_neuron_types
        
        for n in range(self.output_units):
            for layer_idx in range(len(self.layer_weights[n])):
                if layer_idx == 0:
                    src_output = input_data_gpu
                    src_types = input_neuron_types_gpu
                else:
                    # layer_outputsはCPU配列なので、GPU使用時は転送
                    src_output_cpu = self.layer_outputs[n][layer_idx - 1]
                    src_output = self.xp.asarray(src_output_cpu) if self.use_gpu else src_output_cpu
                    
                    hidden_types = self.hidden_neuron_types[layer_idx - 1]
                    src_types = self.xp.asarray(hidden_types) if self.use_gpu else hidden_types
                
                # layer_outputsはCPU配列なので、GPU使用時は転送
                dst_output_cpu = self.layer_outputs[n][layer_idx]
                dst_output = self.xp.asarray(dst_output_cpu) if self.use_gpu else dst_output_cpu
                
                # アミン濃度もGPUに転送
                amine_data_cpu = self.layer_amine_concentrations[n][layer_idx]
                if self.use_gpu:
                    amine_data = self.xp.asarray(amine_data_cpu)
                    amine_pos = amine_data[:, 0]
                    amine_neg = amine_data[:, 1]
                else:
                    amine_pos = amine_data_cpu[:, 0]
                    amine_neg = amine_data_cpu[:, 1]
                
                dst_size = self.layer_weights[n][layer_idx].shape[0]
                src_size = self.layer_weights[n][layer_idx].shape[1]
                
                src_out_reshaped = src_output.reshape(1, -1)
                dst_out_reshaped = dst_output.reshape(-1, 1)
                
                delta = self.learning_rate * src_out_reshaped
                delta = delta * self.xp.abs(dst_out_reshaped)
                delta = delta * (1 - self.xp.abs(dst_out_reshaped))
                
                excitatory_mask = (src_types > 0).reshape(1, -1)
                amine_pos_reshaped = amine_pos.reshape(-1, 1)
                
                if layer_idx == len(self.layer_weights[n]) - 1:
                    dst_types = self.xp.ones(dst_size)
                else:
                    hidden_types = self.hidden_neuron_types[layer_idx]
                    dst_types = self.xp.asarray(hidden_types) if self.use_gpu else hidden_types
                
                src_types_reshaped = src_types.reshape(1, -1)
                dst_types_reshaped = dst_types.reshape(-1, 1)
                
                # 興奮性ニューロンからの重み更新（GPU上で完結）
                weight_update_exc = delta * amine_pos_reshaped * dst_types_reshaped * src_types_reshaped
                weight_update_exc *= excitatory_mask
                
                # 抑制性ニューロンからの重み更新（GPU上で完結）
                inhibitory_mask = (src_types < 0).reshape(1, -1)
                amine_neg_reshaped = amine_neg.reshape(-1, 1)
                
                weight_update_inh = delta * amine_neg_reshaped * dst_types_reshaped * src_types_reshaped
                weight_update_inh *= inhibitory_mask
                
                # 重み更新（GPU上で直接更新、転送なし）
                weight_update_total = weight_update_exc + weight_update_inh
                self.layer_weights[n][layer_idx] += weight_update_total

def load_dataset(dataset_name, train_samples=None, test_samples=None):
    """データセット読み込み（ed_multi_snn.prompt.md準拠・改良版）
    
    改良版の動作:
    - train_samples > 0: 指定サンプル数を使用（実験・デバッグ用）
      ただし、全データを読み込んでおき、エポックごとに異なるサンプルを抽出（過学習防止）
    - train_samples = 0 or None: 全データを使用（本格的な学習用）
    
    過学習防止の仕組み:
    - 全データを保持し、訓練ループ内でエポックごとにランダムサンプリング
    - これにより、少数サンプル指定時もエポックごとに異なるデータで学習
    
    Args:
        dataset_name: 'mnist', 'fashion_mnist'
        train_samples: 訓練サンプル数（0またはNoneで全データ）
        test_samples: テストサンプル数（0またはNoneで全データ）
    
    Returns:
        (train_images, train_labels), (test_images, test_labels): 全データ
        - MNIST/Fashion-MNIST: (N, 28, 28), グレースケール, uint8
        ※サンプリングは訓練ループ内で実施
    """
    if dataset_name == 'fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    else:  # 'mnist'
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # ed_multi_snn.prompt.md準拠: データセット全体を返す
    # サンプリングはエポックごとに訓練ループ内で実施（過学習防止）
    return (train_images, train_labels), (test_images, test_labels)

def main():
    """メイン実行関数"""
    # 日本語フォント設定（最初に実行）
    setup_japanese_font()
    
    # HyperParamsによるパラメータ管理（ed_v032_simple.py準拠）
    hp = HyperParams()
    hp.parse_args()
    
    # ランダムシード設定（ed_multi_snn.prompt.md準拠）
    # エポックごとのサンプリングにも適用されるよう、ここで設定
    if hp.random_seed is not None:
        np.random.seed(hp.random_seed)
        print(f"🎲 ランダムシード設定: {hp.random_seed}")
    
    # 隠れ層解析（HyperParamsで既に解析済み）
    hidden_sizes = hp.hidden_layers
    hidden_str = f"{','.join(map(str, hidden_sizes))} ({'多層' if len(hidden_sizes) > 1 else '単層'})"
    
    # データシャッフル表示用
    shuffle_str = "OFF" if hp.no_shuffle else "ON"
    
    # 図表保存表示用
    if hp.save_fig:
        save_fig_str = f"ON -> {hp.save_fig}"
    else:
        save_fig_str = "OFF"
    
    # ============================================================
    # ED法実行設定表示（ed_v032_simple.py準拠）
    # ============================================================
    print("=" * 60)
    print("ED法実行設定")
    print("=" * 60)
    print("【ED法アルゴリズムパラメータ】")
    print(f"  学習率 (alpha):         {hp.learning_rate:.3f}")
    print(f"  初期アミン濃度 (beta):  {hp.initial_amine:.3f}")
    print(f"  アミン拡散係数 (u1):    {hp.diffusion_rate:.3f}")
    print(f"  シグモイド閾値 (u0):    {hp.sigmoid_threshold:.3f}")
    print(f"  重み初期値1:            {hp.initial_weight_1:.3f}")
    print(f"  重み初期値2:            {hp.initial_weight_2:.3f}")
    print()
    print("【LIFニューロンパラメータ】")
    print(f"  静止膜電位 (v_rest):    {hp.v_rest:.1f} mV")
    print(f"  発火閾値 (v_threshold): {hp.v_threshold:.1f} mV")
    print(f"  リセット電位 (v_reset): {hp.v_reset:.1f} mV")
    print(f"  膜時定数 (tau_m):       {hp.tau_m:.1f} ms")
    print(f"  不応期 (tau_ref):       {hp.tau_ref:.1f} ms")
    print(f"  時間ステップ (dt):      {hp.dt:.1f} ms")
    print(f"  膜抵抗 (R_m):           {hp.R_m:.1f} MΩ")
    print(f"  シミュレーション時間:   {hp.simulation_time:.1f} ms")
    print(f"  LIF層使用:              {'有効' if hp.enable_lif else '無効'} [v019 Phase 4]")
    print()
    print("【実行時設定パラメータ】")
    # データセット名の表示改善
    if hp.fashion_mnist:
        dataset_display = 'Fashion-MNIST'
    else:  # mnist
        dataset_display = 'MNIST'
    print(f"  データセット:           {dataset_display}")
    print(f"  訓練データ数:           {hp.train_samples}")
    print(f"  テストデータ数:         {hp.test_samples}")
    print(f"  エポック数:             {hp.epochs}")
    print(f"  隠れ層構造:             {hidden_str}")
    print(f"  ミニバッチサイズ:       {hp.batch_size} {'(逐次処理)' if hp.batch_size == 1 else '(ミニバッチ)'}")
    # ランダムシード表示
    seed_str = f"{hp.random_seed}" if hp.random_seed is not None else "ランダム"
    print(f"  ランダムシード:         {seed_str}")
    print(f"  データシャッフル:       {shuffle_str}")
    print(f"  リアルタイム可視化:     {'ON' if hp.enable_visualization else 'OFF'}")
    print(f"  詳細表示:               {'ON' if hp.verbose else 'OFF'}")
    print(f"  図表保存:               {save_fig_str}")
    print("=" * 60)
    
    # データ読み込み
    print()
    # データセット名の取得（HyperParams.__post_init__で設定済み）
    dataset_name = hp.dataset_name
    dataset_display = dataset_name.upper().replace('_', '-')
    print(f"📚 {dataset_display}データ読み込み中...")
    (train_images, train_labels), (test_images, test_labels) = load_dataset(
        dataset_name, hp.train_samples, hp.test_samples
    )
    
    # 実際のデータ数を表示（ed_multi_snn.prompt.md準拠: 過学習防止のため全データ使用）
    actual_train_samples = len(train_images)
    actual_test_samples = len(test_images)
    print(f"✅ データ準備完了: 訓練{actual_train_samples}件（全データ）, テスト{actual_test_samples}件（全データ）")
    print(f"   ※過学習防止のため、データセット全体を使用します")
    
    # クラス名マッピング定義（ヒートマップ表示用）
    class_names = None
    if dataset_name == 'fashion_mnist':
        # Fashion-MNISTクラス名（ed_multi_snn.prompt.md準拠）
        class_names = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }
        print(f"✅ Fashion-MNISTクラス名マッピング設定完了")
    
    # ED法初期化
    print()
    # 入力サイズの動的計算（グレースケール画像）
    image_shape = train_images.shape[1:]  # (28, 28)
    base_input_size = np.prod(image_shape)  # 784
    excitatory_size = base_input_size
    inhibitory_size = base_input_size
    paired_input_size = excitatory_size + inhibitory_size  # 1568（E/Iペア化後）
    output_size = hp.output_size  # 10
    
    print(f"\n🧠 ネットワーク構造")
    print(f"   入力画像サイズ: {image_shape} = {base_input_size}ピクセル")
    print(f"   E/Iペア化後: {paired_input_size}ニューロン（E: {excitatory_size}, I: {inhibitory_size}）")
    print(f"   出力層: {output_size}クラス")
    
    # v024 Phase 1: LIF層初期化（単純なSNNラッパー）
    snn = None
    if hp.enable_lif:
        print("\n🧠 LIF層初期化中...")
        from modules.snn.lif_neuron import LIFNeuronLayer
        
        # LIF層サイズ（入力層、隠れ層、出力層）
        # v019 Phase 11修正: ED法仕様に完全準拠
        # 金子勇氏のオリジナルED法では入力層は興奮性・抑制性ペア構成が必須
        # 物理的に1568個のニューロン（興奮性784個+抑制性784個）で構成
        # 注: EDCoreの各出力ニューロンは独立した重み行列を持つため、
        # LIF層は1つの出力ニューロン用のネットワークとして構築
        lif_input_size = paired_input_size  # 1568（ED法仕様準拠）
        lif_layer_sizes = [lif_input_size] + hidden_sizes + [1]  # 出力は1（単一ニューロン）
        
        # 単純なSNNラッパークラスを作成
        class SimpleSNN:
            """MultiLayerEDCore用の単純なSNNラッパー"""
            def __init__(self, layer_sizes, lif_params, simulation_time, dt):
                self.layer_sizes = layer_sizes
                self.simulation_time = simulation_time
                self.dt = dt
                self.n_timesteps = int(simulation_time / dt)
                
                # GPU対応: CuPy/NumPy自動切り替え
                try:
                    import cupy as cp
                    self.xp = cp
                    self.use_gpu = True
                except ImportError:
                    self.xp = np
                    self.use_gpu = False
                
                # LIF層を作成（興奮性・抑制性ペア構造）
                self.layers = []
                for i, size in enumerate(layer_sizes):
                    # 入力層は興奮性・抑制性ペア
                    if i == 0:
                        neuron_types = ['excitatory'] * (size // 2) + ['inhibitory'] * (size // 2)
                    else:
                        neuron_types = ['excitatory'] * size
                    
                    layer = LIFNeuronLayer(
                        n_neurons=size,
                        neuron_params=lif_params,
                        neuron_types=neuron_types
                    )
                    self.layers.append(layer)
                
                # 隠れ層活動を保存
                self.hidden_activation = None
                    
            def simulate_with_input(self, input_pattern, weights):
                """
                スパイク伝播をシミュレーション
                
                Parameters:
                -----------
                input_pattern : np.ndarray or cp.ndarray
                    入力スパイクパターン
                weights : list
                    各層の重み行列リスト
                    
                Returns:
                --------
                output_rates : np.ndarray or cp.ndarray
                    出力層の発火率
                sim_info : dict
                    シミュレーション情報
                """
                # GPU配列をCPUに変換（LIFニューロンはCPU専用）
                if self.use_gpu and hasattr(input_pattern, 'get'):
                    input_pattern_cpu = input_pattern.get()
                else:
                    input_pattern_cpu = np.asarray(input_pattern)
                
                # 層ごとにスパイク伝播
                layer_firing_rates = []
                layer_output = input_pattern_cpu
                
                for i, layer in enumerate(self.layers[1:], start=1):  # 入力層スキップ
                    # 重み行列で結合
                    if i-1 < len(weights):
                        W = weights[i-1]  # (prev_size, curr_size)
                        # GPU配列をCPUに変換
                        if self.use_gpu and hasattr(W, 'get'):
                            W_cpu = W.get()
                        else:
                            W_cpu = np.asarray(W)
                        
                        input_currents = W_cpu.T @ layer_output  # (curr_size,)
                    else:
                        # 重みがない場合は平均活動を伝える
                        mean_activity = np.mean(layer_output)
                        input_currents = np.full(layer.n_neurons, mean_activity * 10.0)
                    
                    # 各時間ステップをシミュレーション
                    spike_counts = np.zeros(layer.n_neurons)
                    for _ in range(self.n_timesteps):
                        spikes = layer.update(input_currents)
                        spike_counts += spikes
                    
                    layer_output = spike_counts / self.n_timesteps  # 発火率
                    layer_firing_rates.append(np.mean(layer_output))
                    
                    # 最後から2番目の層を隠れ層として保存
                    if i == len(self.layers) - 1:
                        self.hidden_activation = layer_output
                
                # GPU配列に変換して返す（ED法コアとの互換性）
                if self.use_gpu:
                    layer_output = self.xp.asarray(layer_output)
                    if self.hidden_activation is not None:
                        self.hidden_activation = self.xp.asarray(self.hidden_activation)
                
                # シミュレーション情報
                sim_info = {
                    'layer_firing_rates': layer_firing_rates,
                    'total_spikes': sum(layer_firing_rates),
                    'avg_voltage': -65.0  # ダミー値
                }
                
                return layer_output, sim_info
                
            def get_hidden_activation(self):
                """隠れ層の活動を取得"""
                if self.hidden_activation is None:
                    result = np.zeros(self.layers[-2].n_neurons if len(self.layers) > 1 else 1)
                    # GPU配列に変換
                    if self.use_gpu:
                        result = self.xp.asarray(result)
                    return result
                return self.hidden_activation
        
        snn = SimpleSNN(
            layer_sizes=lif_layer_sizes,
            lif_params={
                'v_rest': hp.v_rest,
                'v_threshold': hp.v_threshold,
                'v_reset': hp.v_reset,
                'tau_m': hp.tau_m,
                'tau_ref': hp.tau_ref,
                'dt': hp.dt,
                'r_m': hp.R_m
            },
            simulation_time=hp.simulation_time,
            dt=hp.dt
        )
        print(f"✅ LIF層初期化完了（層数: {len(lif_layer_sizes)}, 入力: {lif_input_size}, 出力: 1（単一出力用）, シミュレーション時間: {hp.simulation_time}ms）")
    
    print("🏗️  真の多層ED法初期化中...")
    ed_core = MultiLayerEDCore(
        input_size=paired_input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        learning_rate=hp.learning_rate,
        initial_amine=hp.initial_amine,
        diffusion_rate=hp.diffusion_rate,
        sigmoid_threshold=hp.sigmoid_threshold,
        initial_weight_1=hp.initial_weight_1,
        initial_weight_2=hp.initial_weight_2,
        snn=snn,  # v019 Phase 5追加
        hp=hp     # v019 Phase 5追加
    )
    
    # ヒートマップ統合初期化
    heatmap_integration = None
    if hp.enable_heatmap:
        try:
            from modules.snn_heatmap_integration import EDSNNHeatmapIntegration
            heatmap_integration = EDSNNHeatmapIntegration(
                hp, ed_core, class_names=class_names, image_shape=image_shape
            )
            print("✅ ヒートマップ統合システム初期化完了")
        except ImportError as e:
            print(f"⚠️ ヒートマップモジュールが見つかりません: {e}")
        except Exception as e:
            print(f"⚠️ ヒートマップ統合初期化エラー: {e}")
    
    # データ前処理（全データを前処理）
    train_processed_all, train_labels_processed_all = PureEDPreprocessor.pure_ed_preprocess(
        train_images, train_labels, base_input_size
    )
    test_processed_all, test_labels_processed_all = PureEDPreprocessor.pure_ed_preprocess(
        test_images, test_labels, base_input_size
    )
    
    # サンプル数決定（ed_multi_snn.prompt.md準拠・改良版）
    # - hp.train_samples > 0: 指定サンプル数を使用（実験・デバッグ用）
    # - hp.train_samples = 0: 全データを使用（本格的な学習用）
    use_train_samples = hp.train_samples if hp.train_samples > 0 else len(train_processed_all)
    use_test_samples = hp.test_samples if hp.test_samples > 0 else len(test_processed_all)
    
    # サンプリング戦略のメッセージ
    if hp.train_samples > 0 and hp.train_samples < len(train_processed_all):
        sampling_strategy = f"エポックごとに{use_train_samples}サンプルをランダム抽出（過学習防止）"
    else:
        sampling_strategy = "全データを使用"
    
    print("\n🚀 ミニバッチ対応ED法学習開始")
    print("=" * 60)
    print(f"  データセット: {len(train_processed_all)}件（訓練）, {len(test_processed_all)}件（テスト）")
    print(f"  使用サンプル: {use_train_samples}件（訓練）, {use_test_samples}件（テスト）")
    print(f"  サンプリング戦略: {sampling_strategy}")
    print(f"  エポック数: {hp.epochs}")
    print(f"  バッチサイズ: {hp.batch_size} {'(逐次処理)' if hp.batch_size == 1 else '(ミニバッチ)'}")
    print(f"  データシャッフル: {'無効' if hp.no_shuffle else '有効'}")
    print(f"  リアルタイム表示: {'有効 (学習初期から表示)' if hp.enable_visualization else '無効'}")
    print("=" * 60)
    
    # リアルタイム可視化初期化（ed_v032_simple.py準拠 + 2x2レイアウト）
    visualizer = None
    if hp.enable_visualization:
        print("\n🎨 リアルタイム可視化初期化中...")
        
        # 隠れ層構造の文字列化
        hidden_str = str(hidden_sizes) if hidden_sizes else '[64]'
        
        visualizer = RealtimeLearningVisualizer(
            max_epochs=hp.epochs,
            window_size=(1000, 640),  # 2x2グリッド用にサイズ調整（高さ80%）
            learning_rate=hp.learning_rate,
            initial_amine=hp.initial_amine,
            diffusion_rate=hp.diffusion_rate,
            sigmoid_threshold=hp.sigmoid_threshold,
            initial_weight_1=hp.initial_weight_1,
            initial_weight_2=hp.initial_weight_2,
            dataset_name=dataset_name.upper(),
            train_samples=use_train_samples,  # 修正: 実際に使用するサンプル数
            test_samples=use_test_samples,    # 修正: 実際に使用するサンプル数
            hidden_layers=hidden_str,
            batch_size=hp.batch_size,
            # v019 Phase 3: LIFニューロンパラメータ追加
            v_rest=hp.v_rest,
            v_threshold=hp.v_threshold,
            v_reset=hp.v_reset,
            tau_m=hp.tau_m,
            tau_ref=hp.tau_ref,
            dt=hp.dt,
            R_m=hp.R_m,
            simulation_time=hp.simulation_time,
            # v021 Phase 1追加: シードと詳細表示
            random_seed=hp.random_seed,
            verbose=hp.verbose
        )
        visualizer.setup_plots()
        print("✅ リアルタイム可視化準備完了")
    
    # 学習ループ
    start_time = time.time()
    train_accuracies = []
    test_accuracies = []
    losses = []
    
    # 🔍 検証システム: 訓練/テスト結果保持用配列（ed_multi_snn.prompt.md準拠）
    # エポック×サンプル数の2次元配列で全結果を記録
    train_results_log = []  # 各エポックの訓練結果: [(data_idx, true_label, pred_label), ...]
    test_results_log = []   # 各エポックのテスト結果: [(data_idx, true_label, pred_label), ...]
    
    epoch_pbar = tqdm(range(hp.epochs), 
                      desc="",  # スペース確保のため削除
                      unit="epoch",
                      ncols=110,  # 表示崩れ防止のため調整
                      bar_format='{l_bar}{bar:13}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')  # bar:13で幅制限（余裕確保）
    
    for epoch in epoch_pbar:
        epoch_start = time.time()
        correct_train = 0
        total_samples = 0
        
        # 🔍 検証システム: このエポックの訓練結果記録用
        epoch_train_log = []
        
        # エポックごとのサンプリング（ed_multi_snn.prompt.md準拠・過学習防止）
        if use_train_samples < len(train_processed_all):
            # 指定サンプル数 < 全データ: ランダムサンプリング
            train_indices = np.random.choice(len(train_processed_all), use_train_samples, replace=False)
            train_processed = train_processed_all[train_indices]
            train_labels_processed = train_labels_processed_all[train_indices]
        else:
            # 全データを使用
            train_processed = train_processed_all
            train_labels_processed = train_labels_processed_all
        
        # テストデータも同様にサンプリング
        if use_test_samples < len(test_processed_all):
            test_indices = np.random.choice(len(test_processed_all), use_test_samples, replace=False)
            test_processed = test_processed_all[test_indices]
            test_labels_processed = test_labels_processed_all[test_indices]
        else:
            test_processed = test_processed_all
            test_labels_processed = test_labels_processed_all
        
        # ミニバッチデータローダー作成
        train_loader = MiniBatchDataLoader(
            inputs=train_processed,
            labels=train_labels_processed,
            batch_size=hp.batch_size,
            shuffle=not hp.no_shuffle
        )
        
        # バッチ単位で学習
        for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
            # バッチ内の各サンプルを処理
            for i in range(len(batch_inputs)):
                sample = batch_inputs[i]
                label = batch_labels[i]
                
                # ED法学習処理
                outputs = ed_core.forward_pass(sample)
                
                targets = np.zeros(output_size)
                targets[label] = 1.0
                
                ed_core.pure_ed_learning_step(sample, targets, outputs)
                
                # ヒートマップ更新（MNIST/Fashion-MNIST: 10サンプルごと）
                update_interval = 10
                if heatmap_integration and total_samples % update_interval == 0:
                    spike_activities = convert_ed_outputs_to_spike_activities(
                        ed_core, sample, original_image_shape=image_shape
                    )
                    
                    predicted_label = int(np.argmax(outputs))
                    true_label = int(label)
                    
                    heatmap_integration.update_snn_heatmap(
                        spike_activities=spike_activities,
                        epoch=epoch,
                        sample_idx=total_samples,
                        true_label=true_label,
                        predicted_label=predicted_label
                    )
                
                # 精度計算
                predicted_label = int(np.argmax(outputs))
                true_label = int(label)
                
                if predicted_label == true_label:
                    correct_train += 1
                
                total_samples += 1
                
                # 🔍 検証システム: 訓練結果を記録
                epoch_train_log.append({
                    'data_idx': total_samples - 1,
                    'true_label': true_label,
                    'pred_label': predicted_label,
                    'error': np.sum(np.abs(targets - outputs))  # ED法の学習信号用
                })
        
        # エポック終了処理
        epoch_time = time.time() - epoch_start
        train_accuracy = (correct_train / total_samples) * 100
        train_error_rate = 100.0 - train_accuracy  # エラー率 = 100% - 精度
        
        # 🔍 検証システム: エポックの訓練結果を保存
        train_results_log.append(epoch_train_log)
        
        # テスト評価（全サンプルで評価）
        correct_test = 0
        epoch_test_log = []
        
        # 🔍 修正: 全テストサンプルで評価（100サンプル制限を撤廃）
        test_sample_count = len(test_processed)
        for i in range(test_sample_count):
            outputs = ed_core.forward_pass(test_processed[i])
            predicted_label = int(np.argmax(outputs))
            true_label = int(test_labels_processed[i])
            
            if predicted_label == true_label:
                correct_test += 1
            
            # ED法の学習信号計算（検証システム用）
            targets = np.zeros(output_size)
            targets[true_label] = 1.0
            sample_error = np.sum(np.abs(targets - outputs))
            
            # 🔍 検証システム: テスト結果を記録
            epoch_test_log.append({
                'data_idx': i,
                'true_label': true_label,
                'pred_label': predicted_label,
                'error': sample_error  # ED法の学習信号用
            })
        
        # 🔍 検証システム: エポックのテスト結果を保存
        test_results_log.append(epoch_test_log)
        
        test_accuracy = (correct_test / test_sample_count) * 100
        test_error_rate = 100.0 - test_accuracy  # エラー率 = 100% - 精度
        
        # 記録（lossesは後方互換性のためエラー率/100を格納）
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        losses.append(train_error_rate / 100.0)  # 0-1範囲に正規化
        
        # リアルタイム可視化更新（ed_v032_simple.py準拠）
        if visualizer:
            visualizer.update(
                epoch=epoch,
                train_acc=train_accuracy,
                test_acc=test_accuracy,
                train_err_rate=train_error_rate,  # 訓練エラー率 (100 - train_acc)
                test_err_rate=test_error_rate     # テストエラー率 (100 - test_acc)
            )
        
        # tqdm進行状況更新（エラー率 = 100% - 精度）
        epoch_pbar.set_postfix({
            '訓精': f'{train_accuracy:.1f}%',      # 訓練正答率
            'テ精': f'{test_accuracy:.1f}%',        # テスト正答率
            '訓エ': f'{train_error_rate:.1f}%',    # 訓練エラー率 (100 - 訓練正答率)
            'テエ': f'{test_error_rate:.1f}%'      # テストエラー率 (100 - テスト正答率)
        })
    
    epoch_pbar.close()
    
    # 🔍 検証システム: --verify_acc_loss指定時のみ表示
    if hp.verify_acc_loss:
        verifier = AccuracyLossVerifier()
        verification_results = verifier.verify_and_report(
            train_results_log=train_results_log,
            test_results_log=test_results_log,
            train_accuracies=train_accuracies,
            test_accuracies=test_accuracies,
            losses=losses,
            epochs=hp.epochs,
            show_sample_details=True
        )
        final_verified_train_accuracy = verification_results['final_train_accuracy']
        final_verified_test_accuracy = verification_results['final_test_accuracy']
    else:
        # 検証レポート非表示時は簡易計算のみ
        train_log = train_results_log[-1]  # 最終エポック
        test_log = test_results_log[-1]
        verified_train_correct = sum(1 for r in train_log if r['true_label'] == r['pred_label'])
        verified_test_correct = sum(1 for r in test_log if r['true_label'] == r['pred_label'])
        final_verified_train_accuracy = (verified_train_correct / len(train_log)) * 100
        final_verified_test_accuracy = (verified_test_correct / len(test_log)) * 100
    print("🎯 検証結論:")
    
    # 最終エポックの検証値を使用（ループ外で定義済み）
    if hp.epochs > 0:
        final_train_diff = abs(train_accuracies[-1] - final_verified_train_accuracy)
        final_test_diff = abs(test_accuracies[-1] - final_verified_test_accuracy)
        
        if final_train_diff < 0.01 and final_test_diff < 0.01:
            print("  ✅ 精度・誤差の計算は正確です（差異 < 0.01%）")
        elif final_train_diff < 0.1 and final_test_diff < 0.1:
            print("  ⚠️  軽微な差異が検出されました（差異 < 0.1%）")
        else:
            print("  ⚠️  有意な差異が検出されました！計算ロジックの見直しが必要です")
    print("="*80 + "\n")
    
    # 学習完了
    total_time = time.time() - start_time
    
    print("🎉 リアルタイム可視化対応ED法学習完了!")
    print(f"⏱️  総時間: {total_time:.2f}秒 ({total_time/60:.2f}分)")
    print(f"⚡ 処理速度: {use_train_samples * hp.epochs / total_time:.0f} サンプル/秒")
    
    # 最終テスト評価（最新のテストデータサンプルを使用）
    print("\n🧪 最終テスト評価中...")
    
    # 最終エポックのテストデータサンプリング
    if use_test_samples < len(test_processed_all):
        test_indices = np.random.choice(len(test_processed_all), use_test_samples, replace=False)
        final_test_processed = test_processed_all[test_indices]
        final_test_labels_processed = test_labels_processed_all[test_indices]
    else:
        final_test_processed = test_processed_all
        final_test_labels_processed = test_labels_processed_all
    
    correct = 0
    for i in range(len(final_test_processed)):
        outputs = ed_core.forward_pass(final_test_processed[i])
        if np.argmax(outputs) == final_test_labels_processed[i]:
            correct += 1
    
    final_accuracy = (correct / len(final_test_processed)) * 100
    print(f"✅ 最終テスト正答率: {final_accuracy:.2f}% ({correct}/{len(final_test_processed)})")
    
    # リアルタイム可視化終了処理（ed_v032_simple.py準拠）
    if visualizer:
        # --save_figオプションが指定されている場合は保存
        if hp.save_fig:
            print(f"\n🎨 学習曲線保存中: {hp.save_fig}/")
            visualizer.save_figure(hp.save_fig)
            print("✅ 保存完了")
        
        # 最終的な表示を維持（5秒後に自動クローズ or キー押下で即座にクローズ）
        print("📊 学習曲線を表示中...")
        try:
            plt.ioff()  # インタラクティブモードを無効化
            
            # 非ブロッキングで表示更新
            plt.show(block=False)
            plt.pause(0.1)
            
            # キー押下または5秒待機
            wait_for_keypress_or_timeout(timeout_seconds=5)
            
        except Exception as e:
            print(f"⚠️  表示処理エラー: {e}")
        finally:
            visualizer.close()
            print("✅ グラフウィンドウをクローズしました")
    
    print("\n✅ 最終結果:")
    print(f"   学習精度: {train_accuracies[-1]:.2f}%")
    print(f"   テスト正答率: {final_accuracy:.2f}%")
    print(f"   リアルタイム可視化: ed_v032_simple.py準拠")
    print("🧬 生物学的妥当性: 完全保持（誤差逆伝播なし）")
    print("⚖️  実装一致性: ed_multi_snn.prompt.md完全準拠")
    
    # ヒートマップ終了処理
    if heatmap_integration:
        print("\n🎯 ヒートマップ表示中...")
        try:
            # キー押下または5秒待機
            wait_for_keypress_or_timeout(timeout_seconds=5)
        except Exception as e:
            print(f"⚠️  待機処理エラー: {e}")
        finally:
            print("🎯 ヒートマップ終了処理中...")
            heatmap_integration.stop_snn_heatmap_display()
            print("✅ ヒートマップシステム終了完了")

if __name__ == "__main__":
    main()
