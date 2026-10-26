"""Entry point: python3 -m core.main, or python3 core/main.py."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'core'

from .config import DEFAULT_CONFIG, load_config, write_intrinsics


# 函数说明：在 Linux 创建 Qt 窗口前检查显示环境，缺失时给出操作提示，避免因空 DISPLAY 直接中止。
def check_gui_environment():
    if not sys.platform.startswith('linux'):
        return
    platform = os.environ.get('QT_QPA_PLATFORM', '').split(';')[0].split(':')[0].strip().lower()
    # 显式选择的直接显示或测试后端不依赖 X11，不自动改为 offscreen 隐藏真实界面。
    if platform and platform != 'xcb':
        if platform.startswith('wayland') and not os.environ.get('WAYLAND_DISPLAY', '').strip():
            raise RuntimeError('已选择 Wayland 后端，但未设置 WAYLAND_DISPLAY；请从树莓派图形桌面内的终端启动。')
        return
    if os.environ.get('DISPLAY', '').strip():
        return
    message = (
        '未找到 X11 显示环境（DISPLAY 为空），无法启动 Qt 界面。'
        '请登录树莓派触摸屏上的图形桌面，在桌面终端执行 python3 -m core.main；'
        '普通 SSH/纯命令行终端不会自动连接本地桌面。'
        '如需通过 SSH 显示在远端电脑，请在电脑运行 X Server 后使用 ssh -X 登录。'
        '只做图片测量可使用 --image 图片路径 --headless。'
    )
    if os.environ.get('WAYLAND_DISPLAY', '').strip():
        message += (' 检测到 Wayland 会话：若已安装匹配 PyQt5 的 Wayland 插件，'
                    '可设置 QT_QPA_PLATFORM=wayland；否则请使用提供 DISPLAY 的桌面 XWayland 会话。')
    raise RuntimeError(message)


# 函数说明：解析启动参数，执行已有标定结果导入、单图无界面测量或 Qt 界面启动，并返回进程退出码。
def main(argv=None):
    parser = argparse.ArgumentParser(description='基于 A4 参考面的离线单目测量')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--image', type=Path, help='使用图片代替相机')
    parser.add_argument('--camera', type=int, help='使用 OpenCV USB 相机索引')
    parser.add_argument('--mode', choices=('basic', 'advanced'), default='basic')
    parser.add_argument('--digit', type=int, choices=range(10), help='指定发挥题编号；省略则测最小正方形')
    parser.add_argument('--headless', action='store_true', help='单张图片测量并输出 JSON，不启动 Qt')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--import-calibration', type=Path, help='导入已有标定结果 JSON 并退出')
    args = parser.parse_args(argv)
    try:
        if args.import_calibration:
            data = json.loads(args.import_calibration.read_text(encoding='utf-8'))
            data = data.get('Camera', data)
            write_intrinsics(data['camera_matrix'], data['distortion_coefficients'],
                             (data['img_width'], data['img_height']), args.config)
            print('内参已写入：'+str(args.config))
            return 0
        config = load_config(args.config)
        if args.digit is not None and args.mode != 'advanced':
            parser.error('--digit 仅用于 --mode advanced')
        if args.image:
            config['Camera'].update(backend='image', source=str(args.image.resolve()))
        elif args.camera is not None:
            config['Camera'].update(backend='opencv', source=args.camera)
        if args.headless:
            if not args.image:
                parser.error('--headless 需要 --image')
            from .Camera import Camera
            from .Base import Base
            from .Advanced import Advanced
            from .session import save_result
            started = time.monotonic()
            camera = Camera(config).open()
            try:
                frame = camera.read()
                engine = Base(config) if args.mode == 'basic' else Advanced(config)
                result = engine.measure(frame, args.digit)
            finally:
                camera.close()
            elapsed = time.monotonic()-started
            if elapsed >= config['measurement']['timeout_s']:
                raise ValueError(f'测量超时 ({elapsed:.2f}s)')
            directory = Path(config['output_dir'])
            if not directory.is_absolute():
                directory = args.config.resolve().parent/directory
            record = save_result(result, directory, args.mode, elapsed)
            print(json.dumps(dict(result.as_dict(), elapsed_s=elapsed, record=str(record)), ensure_ascii=False, indent=2))
            return 0
        check_gui_environment()
        from PyQt5 import QtCore, QtWidgets
        from .QTUI import MainWindow
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        app = QtWidgets.QApplication(sys.argv[:1])
        window = MainWindow(config, args.config)
        window.modeCombo.setCurrentIndex(0 if args.mode == 'basic' else 1)
        if args.digit is not None:
            window.digit_group.button(args.digit).setChecked(True)
        if args.fullscreen or config['ui']['fullscreen']:
            window.showFullScreen()
        else:
            window.show()
        return app.exec_()
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
