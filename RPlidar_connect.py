import serial.tools.list_ports
import serial
from rplidar import RPLidar

def list_serial_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("シリアルポートが見つかりません。デバイスとドライバーを確認してください。")
        return []
    for p in ports:
        print(f"Port: {p.device} - {p.description} - {p.hwid}")
    return ports  # PortInfo オブジェクトのリストを返す

def is_bluetooth(p):
    # Bluetooth 仮想シリアルを除外するための簡易チェック
    return ('BTHENUM' in (p.hwid or '')) or ('Bluetooth' in (p.description or ''))

def try_connect(port_info, baudrates=(115200, 256000, 128000)):
    port = port_info.device
    if is_bluetooth(port_info):
        print(f"スキップ: {port} は Bluetooth デバイスの可能性があります ({port_info.description})")
        return False

    for baud in baudrates:
        print(f"試行: {port} @ {baud}")
        # まず低レベルでシリアルが開けるかを確認（ここでのエラーはドライバ/物理問題を示す）
        try:
            with serial.Serial(port, baudrate=baud, timeout=1) as s:
                pass
        except Exception as e:
            print(f"低レベルシリアルオープン失敗 ({port} @ {baud}): {e}")
            continue

        # 次に RPLidar を試す
        try:
            lidar = RPLidar(port, baudrate=baud, timeout=1)
            info = lidar.get_info()
            print("Lidar接続成功:", info)
            lidar.disconnect()
            return True
        except Exception as e:
            print(f"接続失敗 ({port} @ {baud}):", e)
            try:
                lidar.disconnect()
            except:
                pass
    return False

if __name__ == "__main__":
    ports = list_serial_ports()
    if not ports:
        raise SystemExit(1)

    success = False
    for p in ports:
        if try_connect(p):
            success = True
            break

    if not success:
        print("全ての試行に失敗しました。以下を確認してください:\n"
              "- 正しいCOMポートを使用しているか (Device Manager)\n"
              "- RPLIDARのモデルに合ったボーレートを使っているか (A1:115200 等)\n"
              "- USBシリアルドライバー (CP210x) を再インストールしてみる\n"
              "- 別のUSBケーブル/ポートで試す\n"
              "- 低レベルで開けるか、PuTTY/RealTerm で試してみる")