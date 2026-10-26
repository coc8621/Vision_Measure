"""Single-camera adapters. SDK is loaded only when Hikrobot is selected."""
import ctypes as ct
import importlib
import os
from pathlib import Path
import sys
import cv2
import numpy as np
from .tuning import TUNABLES

CAMERA_NODES = {"exposure_time": "ExposureTime", "gain": "Gain"}


class Camera:
    # 函数说明：保存输入配置并初始化图像、OpenCV 设备和海康 SDK 的资源状态，此时不打开设备。
    def __init__(self, config):
        self.config = config
        self.cap = None
        self.cam = None
        self.sdk = None
        self.handle = self.opened = self.grabbing = self.initialized = False
        self.still = None

    # 函数说明：按 backend 打开图片、OpenCV 视频源或海康相机并返回当前实例，失败时释放已申请资源。
    def open(self):
        c = self.config['Camera']
        backend = c['backend']
        try:
            if backend == 'image':
                self.still = cv2.imread(str(c['source']))
                if self.still is None:
                    raise RuntimeError(f"无法读取图片：{c['source']}")
            elif backend == 'opencv':
                self.cap = cv2.VideoCapture(c['source'])
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c['img_width'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c['img_height'])
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.cap.isOpened():
                    raise RuntimeError('无法打开 OpenCV 相机/视频')
            elif backend == 'hikrobot':
                self._open_hikrobot()
            else:
                raise ValueError(f'未知相机后端：{backend}')
        except Exception:
            self.close()
            raise
        return self

    # 函数说明：检查海康 SDK 返回码，非零时抛出包含操作名称和十六进制错误码的异常。
    @staticmethod
    def _check(ret, operation):
        if ret != 0:
            raise RuntimeError(f'海康 {operation} 失败：0x{ret & 0xffffffff:08x}')

    # 函数说明：加载 MVS SDK、枚举并打开指定相机，设置采集参数后启动连续取流。
    def _open_hikrobot(self):
        h = self.config['hikrobot']
        path = Path(os.environ.get('HIKROBOT_MVIMPORT', h['sdk_path']))
        if not (path/'MvCameraControl_class.py').is_file():
            raise RuntimeError('未找到 MVS Python SDK，请设置 hikrobot.sdk_path 或 HIKROBOT_MVIMPORT')
        sys.path.insert(0, str(path))
        try:
            self.sdk = importlib.import_module('MvCameraControl_class')
        except (ImportError, OSError) as exc:
            raise RuntimeError(f'MVS SDK 加载失败，请检查架构和动态库路径：{exc}') from exc
        s = self.sdk
        if hasattr(s.MvCamera, 'MV_CC_Initialize'):
            self._check(s.MvCamera.MV_CC_Initialize(), '初始化')
            self.initialized = True
        devices = s.MV_CC_DEVICE_INFO_LIST()
        self._check(s.MvCamera.MV_CC_EnumDevices(s.MV_GIGE_DEVICE | s.MV_USB_DEVICE, devices), '枚举设备')
        index = h['device_index']
        if not 0 <= index < devices.nDeviceNum:
            raise RuntimeError(f'未找到海康相机索引 {index}，检测到 {devices.nDeviceNum} 台')
        device = ct.cast(devices.pDeviceInfo[index], ct.POINTER(s.MV_CC_DEVICE_INFO)).contents
        self.cam = s.MvCamera()
        self._check(self.cam.MV_CC_CreateHandle(device), '创建句柄')
        self.handle = True
        self._check(self.cam.MV_CC_OpenDevice(s.MV_ACCESS_Exclusive, 0), '打开设备')
        self.opened = True
        if device.nTLayerType == s.MV_GIGE_DEVICE:
            packet = self.cam.MV_CC_GetOptimalPacketSize()
            if packet > 0:
                self._check(self.cam.MV_CC_SetIntValue('GevSCPSPacketSize', packet), '设置网络包')
        self._check(self.cam.MV_CC_SetEnumValue('TriggerMode', 0), '关闭触发模式')
        for node, value in (('Width', h['width']), ('Height', h['height'])):
            self._check(self.cam.MV_CC_SetIntValue(node, value), f'设置 {node}')
        self._check(self.cam.MV_CC_SetEnumValue('ExposureAuto', 0), '关闭自动曝光')
        self._check(self.cam.MV_CC_SetEnumValue('GainAuto', 0), '关闭自动增益')
        for node, value in (('ExposureTime', h['exposure_time']), ('Gain', h['gain'])):
            self._check(self.cam.MV_CC_SetFloatValue(node, float(value)), f'设置 {node}')
        self._check(self.cam.MV_CC_SetBoolValue('AcquisitionFrameRateEnable', True), '启用帧率控制')
        self._check(self.cam.MV_CC_SetFloatValue('AcquisitionFrameRate', float(h['fps'])), '设置帧率')
        self._check(self.cam.MV_CC_StartGrabbing(), '开始采集')
        self.grabbing = True

    # 函数说明：读取并返回独立的 BGR 图像；海康短时无数据时返回 None，其他采集错误抛出异常。
    def read(self):
        if self.still is not None:
            return self.still.copy()
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError('相机断开或视频已经结束')
            return frame
        if not self.grabbing:
            raise RuntimeError('相机尚未启动')
        s = self.sdk
        frame = s.MV_FRAME_OUT()
        ret = self.cam.MV_CC_GetImageBuffer(frame, 250)
        if ret != 0:
            # MV_E_NODATA: a timeout is recoverable, disconnect errors are not.
            if ret & 0xffffffff == 0x80000007:
                return None
            self._check(ret, '读取图像')
        try:
            info = frame.stFrameInfo
            size = info.nWidth*info.nHeight*3
            data = (ct.c_ubyte*size)()
            # The installed SDK exposes the extended converter; support older SDKs.
            extended = hasattr(s, 'MV_CC_PIXEL_CONVERT_PARAM_EX') and hasattr(self.cam, 'MV_CC_ConvertPixelTypeEx')
            params = s.MV_CC_PIXEL_CONVERT_PARAM_EX() if extended else s.MV_CC_PIXEL_CONVERT_PARAM()
            params.nWidth, params.nHeight = info.nWidth, info.nHeight
            params.pSrcData, params.nSrcDataLen = frame.pBufAddr, info.nFrameLen
            params.enSrcPixelType = info.enPixelType
            params.enDstPixelType = s.PixelType_Gvsp_BGR8_Packed
            params.pDstBuffer, params.nDstBufferSize = data, size
            convert = self.cam.MV_CC_ConvertPixelTypeEx if extended else self.cam.MV_CC_ConvertPixelType
            self._check(convert(params), '转换像素格式')
            return np.frombuffer(data, np.uint8).reshape(info.nHeight, info.nWidth, 3).copy()
        finally:
            self.cam.MV_CC_FreeImageBuffer(frame)

    # 函数说明：查询海康可调节点的实际值与硬件范围，其他输入后端返回明确的不可调状态。
    def tuning_capabilities(self):
        capabilities = {}
        for key, node in CAMERA_NODES.items():
            spec = TUNABLES['hikrobot'][key]
            info = dict(supported=False, value=self.config['hikrobot'][key],
                        minimum=spec[1], maximum=spec[2],
                        message='当前后端仅支持图像算法调试；相机参数调节需要海康 SDK')
            if self.cam is not None and self.opened:
                try:
                    value = self.sdk.MVCC_FLOATVALUE()
                    self._check(self.cam.MV_CC_GetFloatValue(node, value), f'读取 {node}')
                    info.update(supported=True, value=float(value.fCurValue),
                                minimum=max(spec[1], float(value.fMin)),
                                maximum=min(spec[2], float(value.fMax)), message='')
                    if info['minimum'] > info['maximum']:
                        info.update(supported=False, message=f'{node} 的硬件范围与调试范围不兼容')
                except (RuntimeError, AttributeError) as exc:
                    info['message'] = str(exc)
            capabilities[key] = info
        return capabilities

    # 函数说明：在采集线程内设置单个海康节点并回读实际值，失败时恢复原值，不伪报调参成功。
    def set_tuning_parameter(self, key, value):
        if key not in CAMERA_NODES:
            raise ValueError('未知相机调试参数')
        info = self.tuning_capabilities()[key]
        if not info['supported']:
            raise RuntimeError(info['message'])
        if not np.isfinite(value) or not info['minimum'] <= value <= info['maximum']:
            raise ValueError(f"{key} 超出硬件范围 {info['minimum']}～{info['maximum']}")
        node = CAMERA_NODES[key]
        self._check(self.cam.MV_CC_SetFloatValue(node, float(value)), f'设置 {node}')
        try:
            actual = self.sdk.MVCC_FLOATVALUE()
            self._check(self.cam.MV_CC_GetFloatValue(node, actual), f'回读 {node}')
            if not np.isfinite(actual.fCurValue):
                raise RuntimeError(f'{node} 回读值无效')
            return float(actual.fCurValue)
        except Exception as exc:
            rollback = self.cam.MV_CC_SetFloatValue(node, float(info['value']))
            if rollback != 0:
                raise RuntimeError(f'{exc}；原值恢复失败，请重连相机') from exc
            raise

    # 函数说明：停止采集并释放图片、OpenCV 句柄和海康 SDK 资源，支持未完全打开及重复关闭的情况。
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.still = None
        if self.cam is not None:
            if self.grabbing:
                self.cam.MV_CC_StopGrabbing()
                self.grabbing = False
            if self.opened:
                self.cam.MV_CC_CloseDevice()
                self.opened = False
            if self.handle:
                self.cam.MV_CC_DestroyHandle()
                self.handle = False
            self.cam = None
        if self.initialized:
            self.sdk.MvCamera.MV_CC_Finalize()
            self.initialized = False
