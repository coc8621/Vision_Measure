"""Sample stability and timestamped measurement records."""
from datetime import datetime
import json
from pathlib import Path
import uuid
import cv2
import numpy as np


# 函数说明：检查最近指定数量采样的形状、编号及距离尺寸极差，稳定时返回最接近中位数的实际测量帧。
def stable_result(samples, count, spread_mm):
    if len(samples) < count:
        return None
    recent = samples[-count:]
    if len({(r.kind, r.digit) for r in recent}) != 1:
        return None
    values = np.array([[r.distance_mm, r.size_mm] for r in recent])
    if np.any(np.ptp(values, axis=0) > spread_mm):
        return None
    # Return an actual observed frame closest to the robust median.
    median = np.median(values, axis=0)
    return recent[int(np.argmin(np.linalg.norm(values-median, axis=1)))]


# 函数说明：将标注图像和测量元数据保存为带时间戳及唯一后缀的 JPG/JSON 文件，返回记录路径。
def save_result(result, directory, mode, elapsed_s):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone()
    name = stamp.strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    image_path = directory/(name+'.jpg')
    if not cv2.imwrite(str(image_path), result.annotated):
        raise OSError('测量图像保存失败')
    payload = result.as_dict()
    payload.update(timestamp=stamp.isoformat(), mode=mode, elapsed_s=elapsed_s, image=image_path.name)
    path = directory/(name+'.json')
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return path
