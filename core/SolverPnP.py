"""A4 pose and camera-to-marker distance; all internal lengths are millimetres."""
from dataclasses import dataclass
import cv2
import numpy as np
from .config import MeasurementConfigurationError, require_measurement_ready


@dataclass
class Pose:
    distance_mm: float
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float
    yaw_deg: float
    camera_point_mm: np.ndarray


class SolverPnP:
    # 函数说明：保存内参及纸张尺寸配置，供后续位姿与相机距离求解使用。
    def __init__(self, config):
        self.config = config

    # 函数说明：检查画面与标定分辨率的宽高比，按整幅缩放调整内参，返回内参矩阵及畸变系数。
    def camera_parameters(self, image_shape):
        c = self.config['Camera']
        height, width = image_shape[:2]
        # Only full-frame resizing is permitted; changing sensor ROI is not resizing.
        sx, sy = width / c['img_width'], height / c['img_height']
        if abs(sx / sy - 1) > 0.02:
            raise MeasurementConfigurationError('当前画面与标定分辨率宽高比不一致，请检查 Camera.img_width/img_height 及采集 ROI')
        k = np.asarray(c['camera_matrix'], np.float64).reshape(3, 3).copy()
        k[0] *= sx
        k[1] *= sy
        return k, np.asarray(c['distortion_coefficients'], np.float64)

    # 函数说明：利用 A4 四角求解并优化平面位姿，检查重投影误差，再计算相机光心到底边中点的毫米直线距离及纸面法向与光轴的夹角。
    def solve(self, corners, image_shape):
        m = self.config['measurement']
        require_measurement_ready(self.config)
        k, d = self.camera_parameters(image_shape)
        w, h = m['paper_width_mm'], m['paper_height_mm']
        obj = np.array([[-w/2, -h/2, 0], [w/2, -h/2, 0],
                        [w/2, h/2, 0], [-w/2, h/2, 0]], np.float64)
        pts = np.asarray(corners, np.float64).reshape(4, 2)
        ok, rs, ts, _ = cv2.solvePnPGeneric(obj, pts, k, d, flags=cv2.SOLVEPNP_IPPE)
        candidates = []
        if ok:
            for r, t in zip(rs, ts):
                r, t = cv2.solvePnPRefineLM(obj, pts, k, d, r.copy(), t.copy())
                rot, _ = cv2.Rodrigues(r)
                if np.any((obj @ rot.T + t.ravel())[:, 2] <= 0):
                    continue
                projected, _ = cv2.projectPoints(obj, r, t, k, d)
                err = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pts)**2, axis=1))))
                candidates.append((err, r, t, rot))
        if not candidates:
            raise ValueError('A4 平面位姿求解失败')
        err, r, t, rot = min(candidates, key=lambda item: item[0])
        if err > self.config['opencv']['max_reprojection_error_px']:
            raise ValueError(f'纸张角点误差过大 ({err:.1f}px)，请调整光照/位置')
        # Keep the target reference at the paper's bottom midpoint. Everything
        # stays in camera coordinates: X right, Y down, Z forward, units mm.
        marker = rot @ np.array([0, h/2, 0]) + t.ravel()
        if not np.isfinite(marker).all() or marker[2] <= 0:
            raise ValueError('目标的相机坐标无效或位于相机后方')
        distance = float(np.linalg.norm(marker))
        yaw = float(np.degrees(np.arccos(np.clip(abs(rot[2, 2]), 0, 1))))
        return Pose(distance, r, t, err, yaw, marker)
