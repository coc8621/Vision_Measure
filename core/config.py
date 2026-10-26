"""Shared configuration and the interface used by the existing calibration program."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from .tuning import add_tuning_defaults, merge_tuning, validate_tuning

DEFAULT_CONFIG = Path(__file__).with_name('common.json')


# 函数说明：校验内参、畸变、纸张尺寸和测量范围等配置，合法时返回原配置，异常时拒绝继续使用。
def validate(config):
    add_tuning_defaults(config)
    validate_tuning(config)
    c = config['Camera']
    k = np.asarray(c['camera_matrix'], dtype=float).reshape(3, 3)
    d = np.asarray(c['distortion_coefficients'], dtype=float).ravel()
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('相机内参矩阵不合法')
    if len(d) not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
        raise ValueError('畸变系数必须为 4、5、8、12 或 14 个有限数')
    if c['img_width'] <= 0 or c['img_height'] <= 0:
        raise ValueError('必须提供标定时的图像分辨率')
    m = config['measurement']
    if not 0 < m['timeout_s'] < 5 or not 1 <= m['samples'] <= 20:
        raise ValueError('测量超时必须小于 5 秒，采样数必须在 1～20 之间')
    w, h, b = (m[x] for x in ('paper_width_mm', 'paper_height_mm', 'border_mm'))
    if not 0 < b < min(w, h) / 2 or not 0.5 <= config['opencv']['pixels_per_mm'] <= 4:
        raise ValueError('纸张尺寸、边框宽度或校正分辨率不合法')
    for key in ('basic_size_range_mm', 'square_size_range_mm'):
        lo, hi = m[key]
        if not 0 < lo < hi:
            raise ValueError(f'{key} 范围不合法')
    return config


# 函数说明：以 UTF-8 读取标准 JSON 配置并执行参数校验，返回可供各模块使用的配置字典。
def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding='utf-8') as f:
        return validate(json.load(f))


# 函数说明：接收已有标定的内参、畸变和图像尺寸，校验后原子写回配置并标记已标定，保留其他参数及说明。
def write_intrinsics(camera_matrix, distortion_coefficients, image_size, path=DEFAULT_CONFIG):
    """Call after calibration: image_size=(width,height), no calibration is performed here.

    Intrinsics refer to the full frame at this size. A cropped sensor ROI needs its
    own adjusted principal point; resolution scaling alone does not correct a crop.
    """
    path = Path(path)
    config = load_config(path)
    config['Camera'].update(
        camera_matrix=np.asarray(camera_matrix, dtype=float).reshape(3, 3).ravel().tolist(),
        distortion_coefficients=np.asarray(distortion_coefficients, dtype=float).ravel().tolist(),
        img_width=int(image_size[0]), img_height=int(image_size[1]), calibrated=True,
    )
    save_config(config, path)


# 函数说明：校验完整配置后原子保存标准 JSON，保留中文参数说明，写入失败时清理临时文件。
def save_config(config, path=DEFAULT_CONFIG):
    path = Path(path)
    validate(config)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as f:
            name = f.name
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


# 函数说明：只把已生效调试参数合并到磁盘最新配置，保留外部标定程序写入的内参及其他字段。
def save_tuning(patch, path=DEFAULT_CONFIG):
    current = load_config(path)
    candidate = merge_tuning(current, patch)
    save_config(candidate, path)


class MeasurementConfigurationError(ValueError):
    """A persistent measurement configuration error that must not be retried as detection noise."""


# 函数说明：在采样前检查标定确认状态，配置未完成时给出具体字段提示，不等待测量超时。
def require_measurement_ready(config):
    if not config['Camera'].get('calibrated'):
        raise MeasurementConfigurationError(
            'Camera.calibrated=false：已有内参数值尚未启用；请导入已有标定结果，'
            '或确认 camera_matrix、畸变系数与分辨率后将 calibrated 设为 true。')



# 函数说明：从磁盘重新读取内参和纸张尺寸，保留当前输入后端及未保存调试值，返回通过检查的配置副本。
def refresh_measurement_config(config, path=None):
    candidate = deepcopy(config)
    if path is not None:
        try:
            current = load_config(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MeasurementConfigurationError(f'无法读取测量配置 {path}：{exc}') from exc
        for key in ('camera_matrix', 'distortion_coefficients', 'img_width', 'img_height', 'calibrated'):
            candidate['Camera'][key] = deepcopy(current['Camera'][key])
        for key in ('paper_width_mm', 'paper_height_mm', 'border_mm'):
            candidate['measurement'][key] = deepcopy(current['measurement'][key])
    require_measurement_ready(candidate)
    return candidate
