from rplidar import RPLidar
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time, math, threading
import collections
import msvcrt
import random

# --- 日本語フォント設定（Windows の Meiryo / Yu Gothic 等を優先） ---
for fname in ('Meiryo', 'Yu Gothic', 'YuGothic', 'MS Gothic', 'MS PGothic'):
    if any(f.name == fname for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = fname
        break
# 文字化けが続く場合は、上記フォントがシステムに入っているか確認してください。

PORT = 'COM3'
BAUD = 256000
TIMEOUT = 5

# 表示/収集パラメータ
FRAME_MS = 100           # 描画更新間隔（ms）
MAX_POINTS = 3000        # 描画する最大点数
BUFFER_MAX = 20000       # 内部保持する点の最大数（古い点は破棄）
RMAX = 5.0               # 表示半径[m]

# 点サイズ制御（調整はここだけで良い）
POINT_SIZE_SCALE = 0.6   # quality→サイズのスケール
POINT_SIZE_MIN = 2       # 最小サイズ (ポイント単位)
POINT_SIZE_MAX = 12      # 最大サイズ (ポイント単位)
POINT_ALPHA = 0.85       # 透明度（重なりが見やすくなる）

# 共有バッファ & ステータス
points_buf = collections.deque(maxlen=BUFFER_MAX)  # 各要素: (x, y, dist_m, quality, angle_deg)
last_angle = 0.0
stop_event = threading.Event()
angle_lock = threading.Lock()

def reader_thread(lidar):
    global last_angle
    try:
        # iter_scans はスキャン単位で返すためバッファ管理が安定
        for scan in lidar.iter_scans(max_buf_meas=8192):
            if stop_event.is_set():
                break
            # scan は [(quality, angle, distance), ...]
            for quality, angle, distance in scan:
                if stop_event.is_set():
                    break
                if distance > 0 and quality > 0:
                    r = distance / 1000.0
                    if r <= RMAX:
                        ang = math.radians(angle)
                        x = r * math.cos(ang)
                        y = r * math.sin(ang)
                        points_buf.append((x, y, r, quality, angle))
                        with angle_lock:
                            last_angle = angle
    except Exception as e:
        print("reader thread error:", e)

def run_visualizer():
    lidar = RPLidar(PORT, baudrate=BAUD, timeout=TIMEOUT)
    reader = None
    try:
        print("INFO:", lidar.get_info())
        print("HEALTH:", lidar.get_health())
        lidar.start_motor()
        time.sleep(1.0)

        # スレッド開始
        reader = threading.Thread(target=reader_thread, args=(lidar,), daemon=True)
        reader.start()

        # Matplotlib 初期化
        plt.ion()
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-RMAX, RMAX); ax.set_ylim(-RMAX, RMAX)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.grid(True, linestyle='--', alpha=0.3)

        # マイナーティックを無効化（今回の例外回避に有効）
        try:
            ax.minorticks_off()
        except Exception:
            pass

        sc = ax.scatter([], [], c=[], s=[], cmap='viridis', vmin=0, vmax=RMAX, alpha=POINT_ALPHA)
        cb = fig.colorbar(sc, ax=ax); cb.set_label("距離 [m]")
        heading_line, = ax.plot([0,0],[0,0], color='r', linewidth=2)
        heading_arrow_len = RMAX * 0.9
        plt.show(block=False)

        print("計測中... 'q' で停止")

        last_update = time.time()
        while not stop_event.is_set():
            # キー入力で停止
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ('q','Q','\x1b'):
                    print("停止要求を受けました")
                    stop_event.set()
                    break

            # フレーム更新
            if (time.time() - last_update) * 1000.0 >= FRAME_MS:
                N = len(points_buf)
                if N > 0:
                    # サンプリングして描画点数を制限
                    if N > MAX_POINTS:
                        # ランダムサンプリング（見た目の代表性を確保）
                        idx = sorted(random.sample(range(N), MAX_POINTS))
                        data = [points_buf[i] for i in idx]
                    else:
                        data = list(points_buf)
                    arr = np.array(data)
                    xs = arr[:,0]; ys = arr[:,1]; dists = arr[:,2]; quals = arr[:,3].astype(float)

                    # サイズは quality に応じてだが、最大値を制限する
                    norm_quals = quals / (quals.max() if quals.max()>0 else 1.0)
                    sizes = np.clip(norm_quals * 30.0 * POINT_SIZE_SCALE + POINT_SIZE_MIN,
                                    POINT_SIZE_MIN, POINT_SIZE_MAX)

                    # 描画更新は例外を握りつぶして継続（GUI一時状態でのエラー防止）
                    try:
                        sc.set_offsets(np.column_stack((xs, ys)))
                        sc.set_array(dists)
                        sc.set_sizes(sizes)

                        with angle_lock:
                            ang = last_angle
                        rad = math.radians(ang)
                        hx = heading_arrow_len * math.cos(rad); hy = heading_arrow_len * math.sin(rad)
                        heading_line.set_data([0, hx], [0, hy])

                        ax.set_title(f"LIDAR 点群 - 表示点数: {len(data)} - 向き: {ang:.1f}°  (qで停止)")
                        fig.canvas.flush_events()
                        plt.pause(0.001)
                    except Exception as e:
                        # 表示の一時的なエラーは無視して次フレームへ
                        print("描画エラーを無視:", e)

                last_update = time.time()

            # 少し待つ（CPU負荷低減）
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("Ctrl+C 検知, 終了します")
        stop_event.set()
    except Exception as e:
        print("エラー:", e)
        stop_event.set()
    finally:
        stop_event.set()
        try:
            lidar.stop()
        except:
            pass
        try:
            lidar.stop_motor()
        except:
            pass
        try:
            lidar.disconnect()
        except:
            pass
        print("LIDAR 停止・切断完了")

if __name__ == "__main__":
    run_visualizer()