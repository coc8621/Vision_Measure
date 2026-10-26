"""Shared A4 detection, metric rectification, and measurement data types."""
from dataclasses import dataclass, field
import cv2
import numpy as np
from .SolverPnP import SolverPnP
from .config import require_measurement_ready


@dataclass
class Shape:
    kind: str
    size_mm: float
    corners: np.ndarray
    score: float = 1.0
    digit: int = None


@dataclass
class Measurement:
    distance_mm: float
    size_mm: float
    kind: str
    annotated: np.ndarray
    rectified: np.ndarray
    corners: np.ndarray
    reprojection_error_px: float
    yaw_deg: float
    digit: int = None
    shapes: list = field(default_factory=list)
    camera_point_mm: np.ndarray = None

    # 函数说明：将测量数值及候选图形转换为可序列化的字典，排除图像和矩阵等大块数据。
    def as_dict(self):
        return dict(distance_mm=float(self.distance_mm), size_mm=float(self.size_mm),
                    distance_reference='camera_to_paper_bottom_midpoint',
                    camera_point_mm=self.camera_point_mm.tolist() if self.camera_point_mm is not None else None,
                    camera_depth_mm=float(self.camera_point_mm[2]) if self.camera_point_mm is not None else None,
                    kind=self.kind, digit=self.digit, yaw_deg=float(self.yaw_deg),
                    reprojection_error_px=float(self.reprojection_error_px),
                    shapes=[dict(kind=s.kind, size_mm=float(s.size_mm), digit=s.digit,
                                 score=float(s.score)) for s in self.shapes])


# 函数说明：将四个角点按图像中的左上、右上、右下、左下顺序排列，返回浮点坐标数组。
def order_quad(points):
    p = np.asarray(points, np.float32).reshape(4, 2)
    center = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1]-center[1], p[:, 0]-center[0]))]
    return np.roll(p, -np.argmin(p.sum(axis=1)), axis=0)


# 函数说明：按静态模糊配置及可调固定阈值执行反向二值化，使黑色目标成为非零前景。
def binary(image, options=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    options = options or {}
    kernel = int(options.get('blur_kernel', 3))
    gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return cv2.threshold(gray, options.get('binary_threshold', 127), 255, cv2.THRESH_BINARY_INV)[1]


class PaperProcessor:
    # 函数说明：保存纸面处理配置，创建 PnP 求解器，并初始化毫米像素比例和去畸变映射缓存。
    def __init__(self, config):
        self.config = config
        self.solver = SolverPnP(config)
        self.scale = config['opencv']['pixels_per_mm']
        self._maps = {}

    # 函数说明：去畸变并检测完整 A4 边框，优化角点、求解位姿和校正纸面，返回后续测量所需数据。
    def prepare(self, frame):
        require_measurement_ready(self.config)
        k, d = self.solver.camera_parameters(frame.shape)
        # PnP uses original distorted coordinates; rectification uses undistorted
        # pixels, otherwise radial distortion biases the geometry inside the A4.
        key = frame.shape[:2]
        if key not in self._maps:
            self._maps[key] = cv2.initUndistortRectifyMap(k, d, None, k,
                                                       (frame.shape[1], frame.shape[0]), cv2.CV_16SC2)
        clean = cv2.remap(frame, *self._maps[key], interpolation=cv2.INTER_LINEAR)
        mask = binary(clean, self.config['opencv'])
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        m = self.config['measurement']
        w, h = round(m['paper_width_mm']*self.scale), round(m['paper_height_mm']*self.scale)
        dst = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame.shape[0]*frame.shape[1]*self.config['opencv']['min_paper_area_ratio']:
                continue
            poly = cv2.approxPolyDP(contour, self.config['opencv'].get('contour_epsilon_ratio', .018)*cv2.arcLength(contour, True), True)
            if len(poly) != 4 or not cv2.isContourConvex(poly):
                continue
            q = order_quad(poly)
            if np.any(q[:, 0] < 2) or np.any(q[:, 1] < 2) or np.any(q[:, 0] > frame.shape[1]-3) or np.any(q[:, 1] > frame.shape[0]-3):
                continue
            sides = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
            if not 0.22 < (sides[0]+sides[2])/(sides[1]+sides[3]) < 1.5:
                continue
            transform = cv2.getPerspectiveTransform(q, dst)
            warped = cv2.warpPerspective(mask, transform, (w, h), flags=cv2.INTER_NEAREST)
            b = round(m['border_mm']*self.scale*0.7)
            ring = np.ones((h, w), bool)
            ring[b:-b, b:-b] = False
            coverage = np.mean(warped[ring] > 0)
            inset = round((m['border_mm']+3)*self.scale)
            white = np.mean(warped[inset:-inset, inset:-inset] == 0)
            if coverage < self.config['opencv'].get('paper_border_coverage', .82) or white < 0.18:
                continue
            score = coverage * area
            if best is None or score > best[0]:
                best = score, q
        if best is None:
            raise ValueError('未找到完整 A4 黑色边框，请保证四角入镜且边框不被遮挡')
        q = best[1]
        # Subpixel refinement improves pose at the long working distance.
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        refined = cv2.cornerSubPix(gray, q.copy(), (4, 4), (-1, -1),
                                  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 25, 0.02))
        if np.max(np.linalg.norm(refined-q, axis=1)) < 5:
            q = refined
        # Re-distort the undistorted corners for the PnP solver.
        rays = cv2.undistortPoints(q.reshape(-1, 1, 2), k, None).reshape(-1, 2)
        rays = np.column_stack((rays, np.ones(4)))
        distorted, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, d)
        pose = self.solver.solve(distorted, frame.shape)
        transform = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
        paper = cv2.warpPerspective(clean, transform, (w, h))
        mask = binary(paper, self.config['opencv'])
        margin = round((m['border_mm']+2)*self.scale)
        mask[:margin] = mask[-margin:] = 0
        mask[:, :margin] = mask[:, -margin:] = 0
        return clean, paper, mask, q, pose, transform

    # 函数说明：将候选图形映射回去畸变画面绘制标注，并封装所选图形的完整测量结果。
    def result(self, prepared, shapes, selected):
        clean, paper, mask, q, pose, transform = prepared
        annotated = clean.copy()
        cv2.polylines(annotated, [q.astype(int)], True, (70, 210, 80), 2)
        inverse = np.linalg.inv(transform)
        for shape in shapes:
            color = (0, 210, 255) if shape is selected else (210, 150, 60)
            pixels = cv2.perspectiveTransform(shape.corners.astype(np.float32).reshape(1, -1, 2), inverse)[0]
            cv2.polylines(annotated, [pixels.astype(int)], True, color, 2)
            label = f'{shape.kind} {shape.size_mm/10:.2f}cm'
            if shape.digit is not None:
                label += f' #{shape.digit}'
            cv2.putText(annotated, label, tuple(pixels[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        return Measurement(pose.distance_mm, selected.size_mm, selected.kind, annotated,
                           paper, q, pose.reprojection_error_px, pose.yaw_deg,
                           selected.digit, shapes, pose.camera_point_mm)


# 函数说明：生成无需内参确认的调试预览，支持原图、二值图、Canny 边缘和四边形候选叠加。
def debug_preview(frame, config, view):
    if view == 'raw':
        return frame
    options = config['opencv']
    mask = binary(frame, options)
    if view == 'binary':
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = int(options.get('blur_kernel', 3))
    gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    edges = cv2.Canny(gray, options['canny_threshold1'], options['canny_threshold2'])
    if view == 'edges':
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) < frame.shape[0]*frame.shape[1]*options['min_paper_area_ratio']:
            continue
        poly = cv2.approxPolyDP(contour, options.get('contour_epsilon_ratio', .018)*cv2.arcLength(contour, True), True)
        if len(poly) == 4 and cv2.isContourConvex(poly):
            cv2.polylines(result, [poly], True, (40, 220, 110), 2)
    return result
