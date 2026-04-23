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
import os
import socket
import numpy as np

from rplidar import RPLidar
import serial
from serial.tools import list_ports


def setup_console_encoding():
    """Best-effort console encoding setup to reduce mojibake."""
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# --- 設定 ---
LIDAR_PORT = 'COM3' if os.name == 'nt' else '/dev/ttyUSB0'
LIDAR_BAUD = 256000
LIDAR_TIMEOUT = 5

IMU_PORT = 'COM4' if os.name == 'nt' else '/dev/ttyACM0'
IMU_BAUD = 115200

FRAME_MS = 300    # 変更: 更新間隔を長くして描画コストを下げる（ms）
RMAX = 5.0
MAX_POINTS = 1000 # 変更: 表示点数を減らす
BUFFER_MAX = 3000   # 変更: 全体メモリを抑える（さらに下げ）

IMU_CSV = '/home/pi-shin/research/imu_log.csv'
LIDAR_CSV = '/home/pi-shin/research/lidar_with_imu.csv'

# ソケット通信設定
SOCKET_ENABLED = False  # Trueにするとソケット送信を有効化
SOCKET_SERVER_IP = '127.0.0.1'  # サーバーIPアドレス（デフォルトはローカル）
SOCKET_SERVER_PORT = 50000  # サーバーポート

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

# ソケット通信用
socket_lock = threading.Lock()
socket_conn = None  # グローバルソケット接続

def init_socket_connection(server_ip=SOCKET_SERVER_IP, server_port=SOCKET_SERVER_PORT):
    """サーバーへのソケット接続を初期化"""
    global socket_conn
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with socket_lock:
            socket_conn = sock
        print(f"ソケット接続成功: {server_ip}:{server_port}")
        return sock
    except Exception as e:
        print(f"ソケット接続失敗: {e}")
        return None

def send_track_data(track_data):
    """トラッキング結果をJSONでサーバーに送信"""
    global socket_conn
    if not SOCKET_ENABLED or track_data is None:
        return
    try:
        with socket_lock:
            if socket_conn is None:
                return
            sock = socket_conn
        
        for track_id, x, y, vx, vy in track_data:
            data = {
                "t": time.time(),
                "track_id": int(track_id),
                "x": float(x),
                "y": float(y),
                "vx": float(vx),
                "vy": float(vy)
            }
            message = json.dumps(data) + "\n"
            sock.send(message.encode())
    except Exception as e:
        print(f"ソケット送信エラー: {e}")
        with socket_lock:
            socket_conn = None

def close_socket():
    """ソケット接続を閉じる"""
    global socket_conn
    with socket_lock:
        if socket_conn is not None:
            try:
                socket_conn.close()
            except:
                pass
            socket_conn = None

# --- クロスプラットフォーム キーボード入力ヘルパー ---
tty = None
termios = None

if os.name == 'nt':
    import msvcrt

    def _kbhit(timeout=0.0):
        """Windows: コンソール入力があれば True を返す。"""
        return msvcrt.kbhit()

    def _getwch():
        """Windows: 1 文字読み取る。"""
        ch = msvcrt.getwch()
        return ch
else:
    import tty
    import termios

    def _kbhit(timeout=0.0):
        """Linux/Unix: stdin に読み取り可能なデータがあれば True を返す。"""
        dr, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(dr)

    def _getwch():
        """Linux/Unix: stdin から 1 文字読み取る。"""
        return sys.stdin.read(1)

def imu_reader_thread(port=None, baud=IMU_BAUD, out_csv=IMU_CSV):
    if port is None:
        port = IMU_PORT
    if not port:
        print("IMU port is not configured. IMU reader is disabled.")
        return
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

def _prepare_lidar_serial_line(port, baud, timeout=1.0):
    """Low-level serial prep to recover from broken stream sync before RPLidar open."""
    ser = None
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout, dsrdtr=False, rtscts=False)
        try:
            ser.setDTR(False)
            ser.setRTS(False)
        except Exception:
            pass
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        # stop command then reset command (RPLIDAR protocol)
        try:
            ser.write(b'\xA5\x25')
            time.sleep(0.05)
            ser.write(b'\xA5\x40')
            time.sleep(0.05)
        except Exception:
            pass
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass


def open_lidar_fixed(port, baud, timeout=5):
    """固定ポート・固定ボーレートで LIDAR を開く。"""
    print(f"Trying LIDAR on {port} @ {baud}...")
    _prepare_lidar_serial_line(port, baud, timeout=0.5)
    lidar = RPLidar(port, baudrate=baud, timeout=timeout)
    time.sleep(0.5)

    try:
        serial_obj = getattr(lidar, '_serial', None)
        if serial_obj is not None:
            serial_obj.reset_input_buffer()
            serial_obj.reset_output_buffer()
    except Exception:
        pass

    info = lidar.get_info()
    health = lidar.get_health()
    print(f"LIDAR opened. INFO: {info} HEALTH: {health}")
    return lidar

def lidar_reader_thread(port=None, baud=LIDAR_BAUD, timeout=LIDAR_TIMEOUT, out_csv=LIDAR_CSV):
    """LIDAR 読み取りスレッド（Descriptor length mismatch に対して再接続/バッファクリアを行う）"""
    global last_angle
    if port is None:
        port = LIDAR_PORT
    backoff = 0.5
    lidar = None

    while not stop_event.is_set():
        try:
            print(f"Attempting to connect LIDAR on {port}...")
            lidar = open_lidar_fixed(port, baud, timeout=timeout)
            print(f"Starting LIDAR motor...")
            lidar.start_motor()
            time.sleep(1.0)  # モーター起動完全待機

            with open(out_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(['point_t','x','y','dist_m','quality','angle_deg','imu_t','ax','ay','az','pitch','roll'])
                
                scan_count = 0
                for scan in lidar.iter_scans(max_buf_meas=512):
                    if stop_event.is_set():
                        break
                    
                    scan_count += 1
                    if scan_count % 10 == 0:
                        print(f"  Scan {scan_count} received")
                    
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
            print(f"LIDAR read error: {e}")
            try:
                import rplidar
                if isinstance(e, rplidar.RPLidarException):
                    print(f"RPLidarException detected: {e}")
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

            print(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)
            continue

    print("LIDAR thread ending...")
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
    print("Headless mode: press 'q' to stop (Ctrl+C also works).")
    last_print = 0.0
    fd = None
    old_settings = None

    try:
        if os.name != 'nt' and tty is not None and termios is not None:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setraw(fd)

        while not stop_event.is_set():
            if _kbhit(timeout=0.0):
                ch = _getwch()
                if ch in ('q', 'Q', '\x1b', '\x03'):  # q / Q / ESC / Ctrl+C
                    print("\r\nStop requested")
                    stop_event.set()
                    break
            now = time.time()
            if now - last_print >= 1.0:
                with points_lock, imu_lock:
                    n_pts = len(points_buf)
                    n_imu = len(imu_buf)
                qsize = csv_queue.qsize()
                tracks_summary = process_scan_and_tracks(now)
                send_track_data(tracks_summary)  # トラッキング結果をソケットで送信
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
        if os.name != 'nt' and fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_available_port_info():
    """利用可能なシリアルポートの情報を返す。"""
    infos = []
    try:
        for p in list_ports.comports():
            infos.append({
                'device': p.device,
                'description': p.description,
                'hwid': p.hwid,
            })
    except Exception:
        pass
    return infos

def _port_exists(port_name, port_infos):
    if not port_name:
        return False
    target = str(port_name).strip().lower()
    return any(str(p['device']).strip().lower() == target for p in port_infos)

def _format_port_list(port_infos):
    if not port_infos:
        return "(none)"
    lines = []
    for p in port_infos:
        lines.append(f"- {p['device']}: {p['description']} [{p['hwid']}]")
    return "\n".join(lines)

def choose_ports(cli_lidar=None, cli_imu=None):
    """ポート存在確認と簡易自動割当を行う。"""
    port_infos = get_available_port_info()
    lidar_port = cli_lidar if cli_lidar else LIDAR_PORT
    imu_port = cli_imu if cli_imu else IMU_PORT

    lidar_from_default = cli_lidar is None
    imu_from_default = cli_imu is None

    available_devices = [p['device'] for p in port_infos]
    used = set()

    if _port_exists(lidar_port, port_infos):
        used.add(lidar_port)
    if _port_exists(imu_port, port_infos):
        used.add(imu_port)

    if lidar_from_default and not _port_exists(lidar_port, port_infos) and available_devices:
        lidar_port = available_devices[0]
        used.add(lidar_port)
        print(f"LIDAR port auto-selected: {lidar_port}")

    if imu_from_default and not _port_exists(imu_port, port_infos):
        for dev in available_devices:
            if dev not in used:
                imu_port = dev
                used.add(imu_port)
                print(f"IMU port auto-selected: {imu_port}")
                break

    lidar_ok = _port_exists(lidar_port, port_infos)
    imu_ok = _port_exists(imu_port, port_infos)

    if not lidar_ok or not imu_ok:
        print("Available serial ports:")
        print(_format_port_list(port_infos))
        if not lidar_ok:
            print(f"LIDAR port not found: {lidar_port}")
        if not imu_ok:
            print(f"IMU port not found: {imu_port}")

    return lidar_port if lidar_ok else None, imu_port if imu_ok else None

def main():
    global LIDAR_PORT, IMU_PORT, LIDAR_BAUD
    setup_console_encoding()
    cli_lidar = sys.argv[1] if len(sys.argv) >= 2 else None
    cli_imu = sys.argv[2] if len(sys.argv) >= 3 else None
    LIDAR_PORT, IMU_PORT = choose_ports(cli_lidar, cli_imu)

    if not LIDAR_PORT:
        print("LIDAR not found. Please provide correct serial ports via CLI args.")
        print("Example (Windows): python Lidar_IMU_raspi_2.py COM5 COM6")
        print("Example (Raspi): python3 Lidar_IMU_raspi_2.py /dev/ttyUSB0 /dev/ttyACM0")
        return
    if not IMU_PORT:
        print("IMU not found. Continuing with IMU disabled.")

    print(f"Using fixed LIDAR settings: {LIDAR_PORT} @ {LIDAR_BAUD}")

    print("=" * 60)
    print("Startup Info")
    print("=" * 60)
    print(f"LIDAR port: {LIDAR_PORT}")
    print(f"LIDAR baud: {LIDAR_BAUD}")
    print(f"IMU port: {IMU_PORT if IMU_PORT else '(disabled)'}")
    print(f"IMU baud: {IMU_BAUD}")
    print(f"Socket tx: {'on' if SOCKET_ENABLED else 'off'}")
    print("=" * 60)

    # ソケット接続を初期化
    if SOCKET_ENABLED:
        print(f"ソケット接続を試行中: {SOCKET_SERVER_IP}:{SOCKET_SERVER_PORT}")
        init_socket_connection()
    
    t_csv = threading.Thread(target=csv_writer_thread, daemon=True)
    t_csv.start()

    t_imu = None
    if IMU_PORT:
        t_imu = threading.Thread(target=imu_reader_thread, daemon=True)
    t_lidar = threading.Thread(target=lidar_reader_thread, daemon=True)

    if t_imu is not None:
        t_imu.start()
    t_lidar.start()
    try:
        headless_run()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        close_socket()  # ソケット接続を閉じる
        if t_imu is not None:
            t_imu.join(timeout=1)
        t_lidar.join(timeout=1)
        t_csv.join(timeout=1)
        print("Shutdown complete")

if __name__ == "__main__":
    main()
