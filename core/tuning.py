"""Live tuning metadata shared by the UI, worker, validation and persistence."""
from copy import deepcopy
import math

# Camera ranges below are UI fallbacks; connected Hikrobot nodes supply real limits.
PARAMETER_LIMITS = {
    'hikrobot': {
        'exposure_time': ('曝光 (µs)', 1., 1000000., 100., 1),
        'gain': ('增益', 0., 100., .1, 2),
        'fps': ('采集帧率', 1., 240., 1., 1),
    },
    'opencv': {
        'binary_threshold': ('固定阈值', 0, 255, 1, 0),
        'blur_kernel': ('模糊核 (奇数)', 1, 15, 2, 0),
        'canny_threshold1': ('Canny 低阈值', 0, 255, 1, 0),
        'canny_threshold2': ('Canny 高阈值', 0, 255, 1, 0),
        'pixels_per_mm': ('校正密度 (px/mm)', .5, 4., .1, 1),
        'min_paper_area_ratio': ('最小纸面占比', .001, .5, .001, 3),
        'contour_epsilon_ratio': ('轮廓近似系数', .005, .05, .001, 3),
        'paper_border_coverage': ('边框黑色覆盖率', .5, .99, .01, 2),
        'max_reprojection_error_px': ('PnP 误差上限 (px)', .1, 20., .1, 1),
    },
    'measurement': {
        'digit_min_score': ('数字最低得分', 0., 1., .01, 2),
        'digit_score_margin': ('数字得分差', 0., 1., .01, 2),
        'max_spread_mm': ('稳定极差 (mm)', .1, 30., .5, 1),
    },
}

# UI 和动态保存只开放五个参数；其余保留静态配置及有效性检查。
TUNABLES = {
    'hikrobot': {key: PARAMETER_LIMITS['hikrobot'][key] for key in ('exposure_time', 'gain')},
    'opencv': {key: PARAMETER_LIMITS['opencv'][key]
               for key in ('binary_threshold', 'canny_threshold1', 'canny_threshold2')},
}

IMAGE_DEFAULTS = dict(binary_threshold=127, blur_kernel=3,
                      contour_epsilon_ratio=.018, paper_border_coverage=.82)


# 函数说明：统一使用固定阈值二值化，并为旧配置补齐缺少的静态图像参数。
def add_tuning_defaults(config):
    config['opencv']['threshold_mode'] = 'fixed'
    for key, value in IMAGE_DEFAULTS.items():
        config['opencv'].setdefault(key, value)
    return config


# 函数说明：检查动态及静态算法参数的类型、范围及奇数约束，非法参数抛出中文异常。
def validate_tuning(config):
    for section, specs in PARAMETER_LIMITS.items():
        for key, spec in specs.items():
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{spec[0]}必须为有限数值')
            if not spec[1] <= value <= spec[2]:
                raise ValueError(f'{spec[0]}必须在 {spec[1]}～{spec[2]} 之间')
            if spec[4] == 0 and int(value) != value:
                raise ValueError(f'{spec[0]}必须为整数')
            if key == 'blur_kernel' and int(value) % 2 != 1:
                raise ValueError(f'{spec[0]}必须为奇数')
    if config['opencv']['canny_threshold1'] >= config['opencv']['canny_threshold2']:
        raise ValueError('Canny 低阈值必须小于高阈值')


# 函数说明：从完整配置提取允许调试的参数快照，排除相机内参、ROI 和其他静态配置。
def tuning_snapshot(config):
    return {section: {key: deepcopy(config[section][key]) for key in specs}
            for section, specs in TUNABLES.items()}


# 函数说明：将调试补丁合并到配置副本，只允许白名单参数并进行整体调参校验。
def merge_tuning(config, patch):
    candidate = deepcopy(config)
    add_tuning_defaults(candidate)
    for section, values in patch.items():
        if section not in TUNABLES or not isinstance(values, dict):
            raise ValueError('调试参数分组不合法')
        for key, value in values.items():
            if key not in TUNABLES[section]:
                raise ValueError(f'不允许动态修改参数：{section}.{key}')
            candidate[section][key] = value
    validate_tuning(candidate)
    return candidate
