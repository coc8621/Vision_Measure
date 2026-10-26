"""7-inch touch UI. Camera acquisition and measurement run in a worker thread."""
from copy import deepcopy
from pathlib import Path
import queue
import threading
import time
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from .Advanced import Advanced
from .Base import Base
from .Camera import Camera
from .session import save_result, stable_result
from .config import save_tuning, refresh_measurement_config, MeasurementConfigurationError
from .DebugPanel import DebugPanel
from .tuning import merge_tuning, tuning_snapshot
from .vision import debug_preview


# 函数说明：将 OpenCV BGR 图像转换为独立持有像素数据的 RGB QImage，供跨线程预览显示使用。
def qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888).copy()


class CaptureWorker(QtCore.QThread):
    preview = QtCore.pyqtSignal(QtGui.QImage)
    status = QtCore.pyqtSignal(str)
    ready = QtCore.pyqtSignal()
    success = QtCore.pyqtSignal(object, float)
    failure = QtCore.pyqtSignal(str)
    capabilities = QtCore.pyqtSignal(object)
    tuning_applied = QtCore.pyqtSignal(int, object, object, str)

    # 函数说明：保存配置，创建容量为一的测量命令队列及线程停止事件。
    def __init__(self, config, config_path=None):
        super().__init__()
        self.config = deepcopy(config)
        self.config_path = config_path
        self.settings_lock = threading.Lock()
        self.settings_request = None
        self.debug_enabled = False
        self.debug_view = "binary"
        self.commands = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

    # 函数说明：线程安全地提交最新调参快照及预览状态，覆盖尚未执行的旧请求以避免滑动时积压。
    def request_tuning(self, patch, enabled, view, revision):
        with self.settings_lock:
            self.settings_request = (deepcopy(patch), enabled, view, revision)

    # 函数说明：在采集线程应用合法图像参数和相机节点，回报实际生效值与错误，节点失败不终止预览。
    def apply_tuning(self, camera, engines, request):
        patch, self.debug_enabled, self.debug_view, revision = request
        errors = []
        try:
            candidate = merge_tuning(self.config, patch)
            for key, value in patch.get('hikrobot', {}).items():
                if value == self.config['hikrobot'][key]:
                    continue
                try:
                    candidate['hikrobot'][key] = camera.set_tuning_parameter(key, value)
                except (RuntimeError, ValueError) as exc:
                    candidate['hikrobot'][key] = self.config['hikrobot'][key]
                    errors.append(str(exc))
            for section, values in tuning_snapshot(candidate).items():
                self.config[section].update(values)
        except ValueError as exc:
            errors.append(str(exc))
        capabilities = camera.tuning_capabilities()
        for key, info in capabilities.items():
            if info['supported']:
                self.config['hikrobot'][key] = info['value']
        self.tuning_applied.emit(revision, tuning_snapshot(self.config), capabilities, '；'.join(errors))

    # 函数说明：在后台打开相机、持续预览并处理测量请求，检查采样稳定性和超时，退出时释放相机。
    def run(self):
        camera = Camera(self.config)
        pending = None
        try:
            engines = {'basic': Base(self.config), 'advanced': Advanced(self.config)}
            camera.open()
            capabilities = camera.tuning_capabilities()
            for key, info in capabilities.items():
                if info['supported']:
                    self.config['hikrobot'][key] = info['value']
            self.capabilities.emit(capabilities)
            self.tuning_applied.emit(0, tuning_snapshot(self.config), capabilities, '')
            self.ready.emit()
            self.status.emit('相机已连接 · 放置目标后点击一键测量')
            last_frame = time.monotonic()
            while not self.stop_event.is_set():
                if pending is None:
                    with self.settings_lock:
                        request, self.settings_request = self.settings_request, None
                    if request is not None:
                        self.apply_tuning(camera, engines, request)
                if pending is None:
                    try:
                        mode, digit, started = self.commands.get_nowait()
                        pending = (mode, digit, started)
                        samples, error = [], '未检测到目标'
                        try:
                            refreshed = refresh_measurement_config(self.config, self.config_path)
                            self.config['Camera'].update(refreshed['Camera'])
                            self.config['measurement'].update(refreshed['measurement'])
                            for engine in engines.values():
                                engine.processor._maps.clear()
                        except MeasurementConfigurationError as exc:
                            self.failure.emit('测量配置错误：'+str(exc))
                            pending = None
                    except queue.Empty:
                        pass
                frame = camera.read()
                now = time.monotonic()
                if pending and now-pending[2] >= self.config['measurement']['timeout_s']:
                    self.failure.emit('测量超时：'+error)
                    pending = None
                if frame is None:
                    if now-last_frame > 2:
                        raise RuntimeError('连续 2 秒未收到相机图像，请检查连接')
                    continue
                last_frame = now
                display_frame = debug_preview(frame, self.config, self.debug_view) if self.debug_enabled else frame
                self.preview.emit(qimage(display_frame))
                if pending:
                    mode, digit, started = pending
                    result = None
                    try:
                        measured = engines[mode].measure(frame, digit)
                        samples.append(measured)
                        samples = samples[-self.config['measurement']['samples']:]
                        result = stable_result(samples, self.config['measurement']['samples'],
                                               self.config['measurement']['max_spread_mm'])
                        error = '采样尚未稳定，请保持目标静止'
                    except MeasurementConfigurationError as exc:
                        self.failure.emit('测量配置错误：'+str(exc))
                        pending = None
                        continue
                    except ValueError as exc:
                        samples = []
                        error = str(exc)
                    elapsed = time.monotonic()-started
                    if elapsed >= self.config['measurement']['timeout_s']:
                        self.failure.emit('测量超时：'+error)
                        pending = None
                    elif result is not None:
                        self.success.emit(result, elapsed)
                        pending = None
                self.msleep(30)
        except Exception as exc:
            self.failure.emit(str(exc))
        finally:
            camera.close()


class MainWindow(QtWidgets.QMainWindow):
    # 函数说明：加载 Designer 布局，设置触摸屏样式、编号按钮及信号连接，并启动相机工作线程。
    def __init__(self, config, config_path):
        super().__init__()
        self.config, self.config_path = config, Path(config_path)
        uic.loadUi(str(Path(__file__).with_name('QTUI.ui')), self)
        self.resize(config['ui']['width'], config['ui']['height'])
        self.setStyleSheet('''
            QMainWindow, QWidget#centralwidget, QWidget#controlContents {background:#101b2b;color:#e4edf7;}
            QLabel {color:#e4edf7;font-size:16px;}
            QLabel#titleLabel {font-size:23px;font-weight:600;}
            QLabel#previewLabel {background:#090f19;border:1px solid #30425a;border-radius:12px;}
            QLabel#distanceLabel,QLabel#sizeLabel {font-size:26px;color:#62d7c3;font-weight:600;}
            QGroupBox {color:#aebed2;border:1px solid #30425a;border-radius:10px;margin-top:12px;padding-top:14px;font-size:15px;}
            QGroupBox::title {subcontrol-origin:margin;left:12px;}
            QPushButton,QComboBox {min-height:40px;border-radius:8px;border:1px solid #40546f;background:#22344d;color:#e4edf7;font-size:17px;padding:2px 10px;}
            QPushButton:checked {background:#176f76;border-color:#62d7c3;}
            QPushButton:disabled,QComboBox:disabled {color:#768497;background:#1b283b;}
            QPushButton#exitButton {background:#743743;border-color:#b45b6c;}
            QPushButton#measureButton {background:#168e80;font-size:22px;font-weight:600;min-height:58px;}
            QCheckBox {color:#e4edf7;font-size:16px;spacing:6px;}
            QCheckBox::indicator {width:22px;height:22px;}
            QSpinBox,QDoubleSpinBox {color:#e4edf7;background:#22344d;border:1px solid #40546f;border-radius:5px;font-size:15px;min-height:36px;}
            QSpinBox:disabled,QDoubleSpinBox:disabled {color:#768497;background:#1b283b;}
            QSpinBox::up-button,QDoubleSpinBox::up-button {width:24px;}
            QSpinBox::down-button,QDoubleSpinBox::down-button {width:24px;}
            QComboBox QAbstractItemView {background:#22344d;color:#e4edf7;selection-background-color:#176f76;}
        ''')
        self.debugCheckBox = QtWidgets.QCheckBox('适应性调试')
        self.debugCheckBox.setMinimumHeight(44)
        self.headerLayout.insertWidget(2, self.debugCheckBox)
        self.debugPanel = DebugPanel(config)
        self.scrollLayout.insertWidget(0, self.debugPanel)
        self.debugPanel.hide()
        self.debugActions = QtWidgets.QWidget()
        debug_actions = QtWidgets.QHBoxLayout(self.debugActions)
        debug_actions.setContentsMargins(0, 0, 0, 0)
        self.debugPanel.saveButton.setText('保存参数')
        self.debugPanel.saveButton.setToolTip('保存已生效参数到当前 common.json')
        self.debugPanel.restoreButton.setText('恢复进入前')
        debug_actions.addWidget(self.debugPanel.saveButton)
        debug_actions.addWidget(self.debugPanel.restoreButton)
        self.controlLayout.addWidget(self.debugActions)
        self.debugActions.hide()
        self.tuning_revision = 0
        self.tuning_pending = False
        self.debug_original = None
        self.debug_timer = QtCore.QTimer(self)
        self.debug_timer.setSingleShot(True)
        self.debug_timer.setInterval(250)
        self.debug_timer.timeout.connect(self.submit_debug_settings)
        self.debugCheckBox.toggled.connect(self.toggle_debug)
        self.debugPanel.changed.connect(self.queue_debug_update)
        self.debugPanel.save_requested.connect(self.save_debug_settings)
        self.debugPanel.restore_requested.connect(self.restore_debug_settings)
        self.modeCombo.addItem('基础题', 'basic')
        self.modeCombo.addItem('发挥题', 'advanced')
        self.digit_group = QtWidgets.QButtonGroup(self)
        self.digit_group.setExclusive(True)
        for digit in range(10):
            button = QtWidgets.QPushButton(str(digit))
            button.setCheckable(True)
            button.setMinimumSize(44, 40)
            self.digitLayout.addWidget(button, digit//5, digit%5)
            self.digit_group.addButton(button, digit)
        self.autoButton.setCheckable(True)
        self.digit_group.addButton(self.autoButton, -2)
        self.autoButton.setChecked(True)
        self.resultBox.setMinimumHeight(165)
        self.digitBox.setMinimumHeight(185)
        self.detailLabel.setMinimumHeight(48)
        self.powerLabel.setMinimumHeight(28)
        QtWidgets.QScroller.grabGesture(self.controlScroll.viewport(), QtWidgets.QScroller.LeftMouseButtonGesture)
        self.modeCombo.currentIndexChanged.connect(self.selection_changed)
        self.digit_group.buttonClicked.connect(self.selection_changed)
        self.measureButton.clicked.connect(self.measure)
        self.reconnectButton.clicked.connect(self.start_camera)
        self.exitButton.clicked.connect(self.close)
        self.worker = None
        self.busy = False
        self.connected = False
        self.last_image = None
        self.hold_result = False
        self.start_camera()
        self.selection_changed()

    # 函数说明：响应模式或编号变化，更新数字选择区并清空旧结果，恢复后续实时预览。
    def selection_changed(self, *_):
        advanced = self.modeCombo.currentData() == 'advanced'
        debugging = self.debugCheckBox.isChecked()
        self.resultBox.setVisible(not debugging)
        self.powerLabel.setVisible(not debugging)
        self.digitBox.setVisible(advanced and not debugging)
        self.digitBox.setEnabled(advanced and not self.busy)
        self.distanceLabel.setText('— cm')
        self.sizeLabel.setText('— cm')
        self.detailLabel.setText('等待测量')
        self.hold_result = False
        self.statusLabel.setText('调试中 · 参数自动应用，退出调试后可测量' if debugging else '等待测量 · 放置目标后点击一键测量')

    # 函数说明：在没有活动采集线程时创建并启动新线程，绑定预览、成功、失败和结束信号。
    def start_camera(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.connected = False
        self.measureButton.setEnabled(False)
        self.reconnectButton.setEnabled(False)
        self.statusLabel.setText('正在连接相机…')
        self.worker = CaptureWorker(self.config, self.config_path)
        self.worker.preview.connect(self.show_preview)
        self.worker.status.connect(self.statusLabel.setText)
        self.worker.ready.connect(self.camera_ready)
        self.worker.success.connect(self.show_result)
        self.worker.failure.connect(self.show_failure)
        self.worker.capabilities.connect(self.debugPanel.set_capabilities)
        self.worker.tuning_applied.connect(self.debug_settings_applied)
        self.worker.finished.connect(self.camera_stopped)
        self.worker.start()

    # 函数说明：接收相机就绪信号，标记连接成功并启用一键测量按钮。
    def camera_ready(self):
        self.connected = True
        if self.debugCheckBox.isChecked():
            self.queue_debug_update()
        self.set_busy(False)

    # 函数说明：接收线程结束信号，清除连接和忙碌状态，禁用测量并允许重连。
    def camera_stopped(self):
        self.connected = False
        self.debug_timer.stop()
        self.tuning_pending = False
        for field_key, field in self.debugPanel.fields.items():
            if field_key[0] == 'hikrobot':
                field.setEnabled(False)
        self.set_busy(False)
        self.measureButton.setEnabled(False)
        self.reconnectButton.setEnabled(True)

    # 函数说明：统一更新测量忙碌状态、按钮文字以及模式和编号控件的可操作状态。
    def set_busy(self, busy):
        self.busy = busy
        debugging = self.debugCheckBox.isChecked()
        self.modeCombo.setEnabled(not busy and not debugging)
        self.debugCheckBox.setEnabled(not busy)
        self.debugPanel.setEnabled(not busy)
        self.debugPanel.saveButton.setEnabled(self.connected and not self.tuning_pending and not busy)
        self.debugPanel.restoreButton.setEnabled(self.connected and not self.tuning_pending and not busy)
        self.digitBox.setEnabled(not busy and self.modeCombo.currentData() == 'advanced')
        self.measureButton.setEnabled(not busy and self.connected and not debugging and not self.tuning_pending)
        self.measureButton.setText('退出调试后测量' if debugging else ('测量中…' if busy else '一键测量'))

    # 函数说明：检查连接及忙碌状态，将当前模式、可选编号和起始时间提交给后台测量线程。
    def measure(self):
        if self.busy or not self.connected or self.debugCheckBox.isChecked() or self.tuning_pending:
            return
        self.selection_changed()
        self.set_busy(True)
        self.statusLabel.setText('正在采样 · 请保持目标静止')
        digit = self.digit_group.checkedId()
        if self.modeCombo.currentData() == 'basic' or digit == -2:
            digit = None
        self.worker.commands.put_nowait((self.modeCombo.currentData(), digit, time.monotonic()))

    # 函数说明：切换适应性调试，保存进入前快照并更新预览状态，调试期间禁止发起测量。
    def toggle_debug(self, enabled):
        if enabled:
            self.debug_original = deepcopy(self.config)
        self.debugPanel.setVisible(enabled)
        self.debugActions.setVisible(enabled)
        self.measureButton.setVisible(not enabled)
        self.selection_changed()
        self.queue_debug_update()
        self.debug_timer.stop()
        self.submit_debug_settings()

    # 函数说明：合并连续控件变更，短暂延迟后只向采集线程提交最新值，并暂时禁用保存与测量。
    def queue_debug_update(self):
        self.tuning_revision += 1
        self.tuning_pending = True
        self.hold_result = False
        self.debugPanel.feedbackLabel.setText('参数待应用…')
        self.set_busy(self.busy)
        self.debug_timer.start()

    # 函数说明：发送当前控件快照和预览类型，设备离线时明确提示而不伪报参数已生效。
    def submit_debug_settings(self):
        if not self.connected or self.worker is None or not self.worker.isRunning():
            self.debugPanel.feedbackLabel.setText('相机未连接；连接成功后才能应用并保存参数。')
            return
        self.worker.request_tuning(self.debugPanel.values(), self.debugCheckBox.isChecked(),
                                   self.debugPanel.viewCombo.currentData(), self.tuning_revision)

    # 函数说明：只接受最新修订的后台回执，以实际生效值同步 UI 与运行配置，失败时保留错误提示。
    def debug_settings_applied(self, revision, snapshot, capabilities, error):
        if revision != self.tuning_revision:
            return
        for section, values in snapshot.items():
            self.config[section].update(values)
        self.debugPanel.set_capabilities(capabilities)
        self.debugPanel.set_values(self.config)
        self.tuning_pending = False
        self.debugPanel.feedbackLabel.setText(('部分参数未应用：'+error) if error else '参数已生效；点击保存后下次启动继续使用。')
        self.set_busy(self.busy)
        if error:
            self.statusLabel.setText('调试参数错误：'+error)
        elif self.debugCheckBox.isChecked():
            self.statusLabel.setText('调试参数已生效 · 可保存参数，或退出调试后测量')

    # 函数说明：将已确认生效的调参值合并到磁盘最新配置，避免覆盖现有相机内参及其他静态设置。
    def save_debug_settings(self):
        if self.tuning_pending or not self.connected:
            return
        try:
            save_tuning(tuning_snapshot(self.config), self.config_path)
            self.debugPanel.feedbackLabel.setText('已保存到 '+str(self.config_path))
            self.statusLabel.setText('调试参数已保存到 '+str(self.config_path))
        except (OSError, ValueError, KeyError) as exc:
            self.debugPanel.feedbackLabel.setText('保存失败：'+str(exc))
            self.statusLabel.setText('保存失败：'+str(exc))

    # 函数说明：恢复进入本次调试时的参数快照，并通过采集线程重新应用到相机和算法。
    def restore_debug_settings(self):
        if self.debug_original is not None:
            self.debugPanel.set_values(self.debug_original)
            self.queue_debug_update()

    # 函数说明：接收后台预览图像；未保留测量结果时更新当前图像并重绘预览区。
    def show_preview(self, image):
        if not self.hold_result:
            self.last_image = image
            self.paint_preview()

    # 函数说明：将当前图像按预览区域尺寸等比例缩放，并绘制到画面标签。
    def paint_preview(self):
        if self.last_image is not None:
            self.previewLabel.setPixmap(QtGui.QPixmap.fromImage(self.last_image).scaled(
                self.previewLabel.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    # 函数说明：处理窗口尺寸变化，重新缩放预览图像以适配可用显示区域。
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'last_image'):
            self.paint_preview()

    # 函数说明：显示厘米单位的距离和尺寸、识别信息及耗时，保留标注画面并保存测量记录。
    def show_result(self, result, elapsed):
        self.set_busy(False)
        self.distanceLabel.setText(f'{result.distance_mm/10:.2f} cm')
        self.distanceLabel.setToolTip('相机光心到纸张底边中点的三维直线距离')
        self.sizeLabel.setText(f'{result.size_mm/10:.2f} cm')
        names = {'square': '正方形', 'triangle': '等边三角形', 'circle': '圆形直径'}
        label = names[result.kind]
        if result.digit is not None:
            label += f' · 编号 {result.digit}'
        if result.camera_point_mm is not None:
            label += f' · Z {result.camera_point_mm[2]/10:.2f} cm'
        self.detailLabel.setText(f'{label}\n耗时 {elapsed:.2f} s · 重投影 {result.reprojection_error_px:.2f} px')
        self.statusLabel.setText('测量完成 · 结果画面已保留，点击一键测量可重新采样')
        self.hold_result = True
        self.last_image = qimage(result.annotated)
        self.paint_preview()
        try:
            directory = Path(self.config['output_dir'])
            if not directory.is_absolute():
                directory = self.config_path.parent/directory
            save_result(result, directory, self.modeCombo.currentData(), elapsed)
        except (OSError, cv2.error) as exc:
            self.statusLabel.setText(f'测量完成，但记录保存失败：{exc}')

    # 函数说明：处理测量或采集异常，清空旧数值、解除忙碌状态并显示错误原因。
    def show_failure(self, message):
        self.set_busy(False)
        self.distanceLabel.setText('— cm')
        self.sizeLabel.setText('— cm')
        self.detailLabel.setText('测量未完成')
        self.hold_result = False
        self.statusLabel.setText(message)

    # 函数说明：关闭窗口前请求工作线程停止，异步等待相机资源释放后再允许窗口关闭。
    def closeEvent(self, event):
        self.debug_timer.stop()
        self.centralwidget.setEnabled(False)
        self.exitButton.setText('正在退出…')
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop_event.set()
            # Never destroy a running QThread or forcibly terminate the SDK call.
            event.ignore()
            self.statusLabel.setText('正在停止采集并释放相机…')
            QtCore.QTimer.singleShot(150, self.close)
            return
        event.accept()
