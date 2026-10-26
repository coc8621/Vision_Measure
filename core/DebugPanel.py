"""Touch-friendly adaptive tuning controls; no camera calls run on the GUI thread."""
from PyQt5 import QtCore, QtWidgets
from .tuning import TUNABLES, tuning_snapshot


class DebugPanel(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal()
    save_requested = QtCore.pyqtSignal()
    restore_requested = QtCore.pyqtSignal()

    # 函数说明：创建可滚动调参区中的预览选择、相机及算法输入框，并提供保存和恢复入口。
    def __init__(self, config, parent=None):
        super().__init__('适应性调试', parent)
        self.fields = {}
        layout = QtWidgets.QVBoxLayout(self)
        self.viewCombo = QtWidgets.QComboBox()
        for label, value in (('原始画面', 'raw'), ('二值化预览', 'binary'),
                             ('Canny 边缘（辅助观察）', 'edges'), ('四边形候选（非测量结果）', 'contours')):
            self.viewCombo.addItem(label, value)
        self.viewCombo.setCurrentIndex(1)
        layout.addWidget(self.viewCombo)
        self.noticeLabel = QtWidgets.QLabel('相机范围查询中…')
        self.noticeLabel.setWordWrap(True)
        layout.addWidget(self.noticeLabel)
        for section, specs in TUNABLES.items():
            title = {'hikrobot': '相机参数 · 实际回读', 'opencv': '固定二值化与 Canny'}[section]
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            for key, spec in specs.items():
                field = QtWidgets.QDoubleSpinBox() if spec[4] else QtWidgets.QSpinBox()
                field.setRange(spec[1], spec[2])
                field.setSingleStep(spec[3])
                if isinstance(field, QtWidgets.QDoubleSpinBox):
                    field.setDecimals(spec[4])
                field.setKeyboardTracking(False)
                field.setValue(config[section][key])
                field.valueChanged.connect(self.field_changed)
                field.setMinimumHeight(36)
                field.setMaximumWidth(155)
                self.fields[(section, key)] = field
                label = QtWidgets.QLabel(spec[0])
                label.setWordWrap(True)
                form.addRow(label, field)
                if section == 'hikrobot':
                    field.setEnabled(False)
            layout.addWidget(box)
        self.feedbackLabel = QtWidgets.QLabel('调整后约 250ms 自动生效；保存后下次启动继续使用。')
        self.feedbackLabel.setWordWrap(True)
        layout.addWidget(self.feedbackLabel)
        self.saveButton = QtWidgets.QPushButton('保存到 common.json')
        self.restoreButton = QtWidgets.QPushButton('恢复进入调试前参数')
        self.saveButton.clicked.connect(self.save_requested)
        self.restoreButton.clicked.connect(self.restore_requested)
        self.viewCombo.currentIndexChanged.connect(self.field_changed)

    # 函数说明：读取所有可调控件形成参数快照，不包含当前后端不支持的相机节点。
    def values(self):
        values = {}
        for (section, key), field in self.fields.items():
            if section == 'hikrobot' and not field.isEnabled():
                continue
            values.setdefault(section, {})[key] = field.value()
        return values

    # 函数说明：在数值或预览类型改变时通知主窗口提交最新参数。
    def field_changed(self, *_):
        self.changed.emit()

    # 函数说明：按后台确认的实际参数刷新控件，阻断信号避免产生重复调参请求。
    def set_values(self, config):
        for section, values in tuning_snapshot(config).items():
            for key, value in values.items():
                field = self.fields[(section, key)]
                blocker = QtCore.QSignalBlocker(field)
                field.setValue(value)
                del blocker

    # 函数说明：用 SDK 查询的节点范围限制相机控件，不支持的节点保持禁用并显示原因。
    def set_capabilities(self, capabilities):
        messages = []
        for key, info in capabilities.items():
            field = self.fields[('hikrobot', key)]
            blocker = QtCore.QSignalBlocker(field)
            field.setEnabled(info['supported'])
            field.setToolTip(info['message'])
            if info['supported']:
                field.setRange(info['minimum'], info['maximum'])
                field.setValue(info['value'])
            elif info['message'] not in messages:
                messages.append(info['message'])
            del blocker
        self.noticeLabel.setText('海康相机参数实时回读；曝光单位 µs。' if not messages else '；'.join(messages))
