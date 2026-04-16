import time
import math
import threading
import collections
import csv
import json
import sys
import queue
import bisect
import select
import termios
import tty
import numpy as np

from rplidar import RPLidar
import serial

# --- 設定 ---
LIDAR_PORT = '/dev/ttyUSB0'
LIDAR_BAUD = 256000
LIDAR_TIMEOUT = 5

IMU_PORT = '/dev/ttyACM0'
IMU_BAUD = 115200

FRAME_MS = 300    # 変更: 更新間隔を長くして描画コストを下げる（ms）
RMAX = 5.0
MAX_POINTS = 1000 # 変更: 表示点数を減らす
BUFFER_MAX = 3000   # 変更: 全体メモリを抑える（さらに下げ）

IMU_CSV = 'imu_log.csv'
LIDAR_CSV = 'lidar_with_imu.csv'

# --- 共有バッファ & ロック ---
points_buf = collections.deque(maxlen=BUFFER_MAX)   # 各要素: (x,y,dist_m,quality,angle_deg, t_sec)
imu_buf = collections.deque(maxlen=500)             # IMU が高速のため小さめに（古いデータ自動破棄）
points_lock = threading.Lock()
imu_lock = threading.Lock()
stop_event = threading.Event()
last_angle = 0.0
angle_lock = threading.Lock()

# CSV 書き込み用キュー
csv_queue = queue.Queue(maxsize=2000)   # 大きすぎるとメモリ膨張するので縮小

# --- クロスプラットフォーム キーボード入力ヘルパー ---
def _kbhit_linux(timeout=0.0):
    """stdin に読み取り可能なデータがあれば True を返す（ブロックしない）"""
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(dr)

def _getwch_linux():
    """stdin から 1 文字読み取る"""
    return sys.stdin.read(1)

def imu_reader_thread(port=IMU_PORT, baud=IMU_BAUD, out_csv=IMU_CSV):
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print("IMU open error:", e)
        return
    while not stop_event.is_set():
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
        except Exception:
            continue
        if not line:
            continue
        parts = [p.strip() for p in line.split(',') if p.strip()!='']
        if len(parts) == 6:
            try:
                vals = [float(p) for p in parts[:6]]
                t_s = vals[0] if vals[0] > 1e3 else time.time()
                sample = (t_s, vals[1], vals[2], vals[3], vals[4], vals[5])
            except ValueError:
                continue
        elif len(parts) == 5:
            try:
                nums = [float(p) for p in parts[:5]]
            except ValueError:
                continue
            t_s = time.time()
            sample = (t_s, nums[0], nums[1], nums[2], nums[3], nums[4])
        else:
            continue
        with imu_lock:
            imu_buf.append(sample)
    try:
        ser.close()
    except:
        pass

def open_lidar_try_baud(port, baud_list=(256000, 115200), timeout=5):
    """複数ボーレートを順に試し、正常に get_info()/get_health() が取れたものを返す。
    見つからなければ例外を投げる。"""
    last_err = None
    for b in baud_list:
        try:
            lidar = RPLidar(port, baudrate=b, timeout=timeout)
            try:
                info = lidar.get_info()
                health = lidar.get_health()
                print(f"LIDAR opened at {b} baud. INFO: {info} HEALTH: {health}")
                return lidar, b
            except Exception as e:
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
                last_err = e
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot open LIDAR on {port} with any baud: {last_err}")

def lidar_reader_thread(port=LIDAR_PORT, baud=LIDAR_BAUD, timeout=LIDAR_TIMEOUT, out_csv=LIDAR_CSV):
    """LIDAR 読み取りスレッド（Descriptor length mismatch に対して再接続/バッファクリアを行う）"""
    global last_angle
    backoff = 1.0
    lidar = None

    while not stop_event.is_set():
        try:
            lidar, used_baud = open_lidar_try_baud(port, baud_list=(256000, 115200), timeout=timeout)
            lidar.start_motor()
            time.sleep(0.5)

            with open(out_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(['point_t','x','y','dist_m','quality','angle_deg','imu_t','ax','ay','az','pitch','roll'])

                for scan in lidar.iter_scans(max_buf_meas=512):
                    if stop_event.is_set():
                        break
                    try:
                        serial_obj = getattr(lidar, '_serial', None)
                        if serial_obj is not None and hasattr(serial_obj, 'in_waiting') and serial_obj.in_waiting > 4096:
                            try:
                                serial_obj.reset_input_buffer()
                                serial_obj.reset_output_buffer()
                                print("Warning: serial input buffer large -> reset")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    scan_t = time.time()
                    for meas in scan:
                        if stop_event.is_set():
                            break
                        try:
                            quality, angle, distance = meas
                        except Exception:
                            continue
                        if distance <= 0 or quality <= 0:
                            continue
                        r = distance / 1000.0
                        if r > RMAX:
                            continue
                        ang = math.radians(angle)
                        x = r * math.cos(ang); y = r * math.sin(ang)
                        with points_lock:
                            points_buf.append((x, y, r, quality, angle, scan_t))
                        with angle_lock:
                            last_angle = angle

                        with imu_lock:
                            imu_sample = imu_buf[-1] if len(imu_buf) > 0 else None
                        if imu_sample:
                            imu_t, ax, ay, az, pitch, roll = imu_sample
                        else:
                            imu_t = 0.0; ax = ay = az = pitch = roll = 0.0

                    row = [scan_t, x, y, r, quality, angle, imu_t, ax, ay, az, pitch, roll]
                    try:
                        csv_queue.put_nowait(row)
                    except queue.Full:
                        try:
                            _ = csv_queue.get_nowait()
                            csv_queue.put_nowait(row)
                        except queue.Empty:
                            pass

            break

        except Exception as e:
            print("LIDAR read error:", e)
            try:
                import rplidar
                if isinstance(e, rplidar.RPLidarException):
                    try:
                        serial_obj = getattr(lidar, '_serial', None)
                        if serial_obj is not None:
                            try:
                                serial_obj.reset_input_buffer()
                                serial_obj.reset_output_buffer()
                                print("Serial buffers reset.")
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                if lidar is not None:
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
            except Exception:
                pass
            lidar = None

            if stop_event.is_set():
                break

            time.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)
            continue

    stop_event.set()
    try:
        if lidar is not None:
            lidar.stop()
    except:
        pass
    try:
        if lidar is not None:
            lidar.stop_motor()
    except:
        pass
    try:
        if lidar is not None:
            lidar.disconnect()
    except:
        pass

def csv_writer_thread(out_csv=LIDAR_CSV):
    """csv_queue から行を取り出してバッチ書き込みするスレッド"""
    with open(out_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['point_t','x','y','dist_m','quality','angle_deg','imu_t','ax','ay','az','pitch','roll'])
        buffer = []
        last_flush = time.time()
        while not stop_event.is_set() or not csv_queue.empty():
            try:
                row = csv_queue.get(timeout=0.2)
                buffer.append(row)
            except queue.Empty:
                row = None
            if len(buffer) >= 200 or (buffer and (time.time() - last_flush) > 1.0):
                try:
                    writer.writerows(buffer)
                    f.flush()
                except Exception:
                    pass
                buffer.clear()
                last_flush = time.time()
        if buffer:
            try:
                writer.writerows(buffer)
                f.flush()
            except Exception:
                pass

def get_latest_imu(t_sec):
    """imu_buf から時刻 t_sec に最も近いサンプルを返す。"""
    with imu_lock:
        if not imu_buf:
            return None
        lst = list(imu_buf)
    times = [s[0] for s in lst]
    idx = bisect.bisect_left(times, t_sec)
    if idx == 0:
        cand = lst[0]
    elif idx >= len(lst):
        cand = lst[-1]
    else:
        left = lst[idx-1]; right = lst[idx]
        cand = left if abs(left[0]-t_sec) <= abs(right[0]-t_sec) else right
    return cand

def get_recent_scan_points(tol=0.02):
    """最新スキャン（代表時刻に近い点群）を返す。返却: Nx2 array of (x,y) and scan_t"""
    with points_lock:
        if not points_buf:
            return np.zeros((0,2)), None
        times = [p[5] for p in points_buf]
        max_t = max(times)
        pts = [p for p in points_buf if abs(p[5] - max_t) <= tol]
    if not pts:
        return np.zeros((0,2)), None
    arr = np.array([[p[0], p[1]] for p in pts], dtype=float)
    return arr, max_t

def euclidean_cluster(points, eps=0.25, min_pts=3):
    """単純な single-link クラスタ"""
    N = points.shape[0]
    if N == 0:
        return []
    visited = np.zeros(N, dtype=bool)
    clusters = []
    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        members = [i]
        while stack:
            u = stack.pop()
            d2 = np.sum((points - points[u])**2, axis=1)
            neigh = np.where(d2 <= eps*eps)[0]
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    members.append(v)
        if len(members) >= min_pts:
            clusters.append(points[members])
    return clusters

# 軽量カルマン（CVモデル, state: x,y,vx,vy）
class Track:
    def __init__(self, x, y, id):
        self.id = id
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 1.0
        self.last_update = time.time()
        self.misses = 0

    def predict(self, t):
        dt = max(1e-6, t - self.last_update)
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]], dtype=float)
        Q = np.eye(4) * 0.01
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q
        self.last_update = t

    def update(self, zx, zy):
        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        R = np.eye(2) * 0.05
        z = np.array([zx, zy], dtype=float)
        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(4) - K.dot(H)).dot(self.P)
        self.misses = 0
        self.last_update = time.time()

    def predict_future(self, dt_h):
        return (self.x[0] + self.x[2]*dt_h, self.x[1] + self.x[3]*dt_h)

tracks = []
_next_track_id = 1

def manage_tracks(detections, now, gate=0.6, max_misses=5, max_tracks=12):
    global tracks, _next_track_id
    for tr in tracks:
        tr.predict(now)
    if len(detections)==0:
        for tr in tracks:
            tr.misses += 1
        tracks = [t for t in tracks if t.misses <= max_misses]
        return tracks

    dets = np.array(detections)
    M = len(tracks); N = dets.shape[0]
    assigned_det = set()
    assigned_tr = set()
    if M>0:
        dmat = np.zeros((M,N), dtype=float)
        for i,tr in enumerate(tracks):
            pred = tr.x[:2]
            dmat[i,:] = np.linalg.norm(dets - pred.reshape(1,2), axis=1)
        for _ in range(min(M,N)):
            i,j = np.unravel_index(np.argmin(dmat), dmat.shape)
            if dmat[i,j] > gate:
                break
            tracks[i].update(dets[j,0], dets[j,1])
            assigned_det.add(j); assigned_tr.add(i)
            dmat[i,:] = 1e6
            dmat[:,j] = 1e6

    for j in range(N):
        if j in assigned_det:
            continue
        if len(tracks) >= max_tracks:
            break
        x,y = dets[j]
        tracks.append(Track(x,y, _next_track_id)); _next_track_id += 1

    for i,tr in enumerate(tracks):
        if i not in assigned_tr:
            tr.misses += 1
    tracks = [t for t in tracks if t.misses <= max_misses]
    return tracks

def process_scan_and_tracks(now=None):
    """最新スキャンをクラスタ化→重心を検出→トラッキング更新。"""
    if now is None:
        now = time.time()
    pts, scan_t = get_recent_scan_points()
    if pts.size == 0:
        return []
    clusters = euclidean_cluster(pts, eps=0.25, min_pts=3)
    detections = []
    for c in clusters:
        centroid = c.mean(axis=0)
        detections.append(centroid)
    manage_tracks(detections, now)
    return [(t.id, float(t.x[0]), float(t.x[1]), float(t.x[2]), float(t.x[3])) for t in tracks]

def headless_run(poll_interval=0.2):
    """プロットなしで動作。定期的に状態を表示し、'q' キーで終了。"""
    print("ヘッドレスモード: 'q' キーで停止、Ctrl+C でも停止できます。")
    last_print = 0.0

    # stdin を raw モードに切り替えてノンブロッキング入力を有効化
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            if _kbhit_linux(timeout=0.0):
                ch = _getwch_linux()
                if ch in ('q', 'Q', '\x1b', '\x03'):  # q / Q / ESC / Ctrl+C
                    print("\r\n停止要求を受信しました")
                    stop_event.set()
                    break
            now = time.time()
            if now - last_print >= 1.0:
                with points_lock, imu_lock:
                    n_pts = len(points_buf)
                    n_imu = len(imu_buf)
                qsize = csv_queue.qsize()
                tracks_summary = process_scan_and_tracks(now)
                s = f"\r[{time.strftime('%H:%M:%S')}] points={n_pts} imu={n_imu} csv_q={qsize}"
                if tracks_summary:
                    s += " | tracks: " + ", ".join(
                        [f"id{tid}:({x:.2f},{y:.2f}) v({vx:.2f},{vy:.2f})"
                         for tid, x, y, vx, vy in tracks_summary])
                print(s)
                last_print = now
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        # 必ず元のターミナル設定を復元する
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    global LIDAR_PORT, IMU_PORT
    if len(sys.argv) >= 2:
        LIDAR_PORT = sys.argv[1]
    if len(sys.argv) >= 3:
        IMU_PORT = sys.argv[2]

    t_csv = threading.Thread(target=csv_writer_thread, daemon=True)
    t_csv.start()

    t_imu = threading.Thread(target=imu_reader_thread, daemon=True)
    t_lidar = threading.Thread(target=lidar_reader_thread, daemon=True)

    t_imu.start()
    t_lidar.start()
    try:
        headless_run()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        t_imu.join(timeout=1)
        t_lidar.join(timeout=1)
        t_csv.join(timeout=1)
        print("終了処理完了")

if __name__ == "__main__":
    main()
