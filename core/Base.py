"""Basic task: circle diameter, equilateral triangle side, square side."""
import cv2
import numpy as np
from .vision import PaperProcessor, Shape


class Base:
    # 函数说明：保存基础题配置并创建 A4 检测、位姿求解和透视校正处理器。
    def __init__(self, config):
        self.processor = PaperProcessor(config)
        self.config = config

    # 函数说明：识别单帧中唯一的圆、等边三角形或正方形，返回距离与尺寸；digit 为统一接口保留参数。
    def measure(self, frame, digit=None):
        prepared = self.processor.prepare(frame)
        mask = prepared[2]
        scale = self.processor.scale
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        lo, hi = self.config['measurement']['basic_size_range_mm']
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (lo*scale)**2 * 0.25:
                continue
            perimeter = cv2.arcLength(contour, True)
            poly = cv2.approxPolyDP(contour, .022*perimeter, True).reshape(-1, 2)
            kind, size = None, None
            if len(poly) == 3:
                sides = np.linalg.norm(np.roll(poly, -1, axis=0)-poly, axis=1)
                if max(sides)/min(sides) < 1.13:
                    kind, size = 'triangle', float(np.mean(sides)/scale)
            elif len(poly) == 4:
                sides = np.linalg.norm(np.roll(poly, -1, axis=0)-poly, axis=1)
                a = np.roll(poly, -1, axis=0)-poly
                cosines = np.abs(np.sum(a*np.roll(a, 1, axis=0), axis=1)/(sides*np.roll(sides, 1)))
                if max(sides)/min(sides) < 1.12 and max(cosines) < .14:
                    rect = cv2.minAreaRect(contour)
                    kind, size = 'square', float(np.mean(rect[1])/scale)
                    poly = cv2.boxPoints(rect)
            elif perimeter > 0 and 4*np.pi*area/perimeter**2 > .84 and len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                if max(ellipse[1])/min(ellipse[1]) < 1.10:
                    kind, size = 'circle', float(np.mean(ellipse[1])/scale)
                    poly = cv2.boxPoints(cv2.minAreaRect(contour))
            if kind and lo-3 <= size <= hi+3:
                shapes.append(Shape(kind, size, poly))
        if len(shapes) != 1:
            raise ValueError('未识别到唯一基本图形（圆/等边三角形/正方形），请检查尺寸和边框')
        return self.processor.result(prepared, shapes, shapes[0])
