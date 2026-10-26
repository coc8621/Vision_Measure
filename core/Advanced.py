"""Square reconstruction from perpendicular visible edges, plus offline digits."""
from itertools import combinations
from pathlib import Path
import cv2
import numpy as np
from .vision import PaperProcessor, Shape, order_quad
from .Advanced_Geometry import (filled_geometry, orthogonal_directions, convex_corner,
                                contour_squares, refine_square)

# 函数说明：裁剪数字前景并等比例缩放、居中到 40×56 画布，返回用于模板比较的二值字形。
def normalize_glyph(glyph):
    pts = cv2.findNonZero(glyph)
    result = np.zeros((56, 40), np.uint8)
    if pts is None:
        return result
    x, y, w, h = cv2.boundingRect(pts)
    scale = min(34/w, 50/h)
    resized = cv2.resize(glyph[y:y+h, x:x+w], (max(1, round(w*scale)), max(1, round(h*scale))))
    rh, rw = resized.shape
    result[(56-rh)//2:(56-rh)//2+rh, (40-rw)//2:(40-rw)//2+rw] = resized
    return (result > 100).astype(np.uint8)


class DigitRecognizer:
    """Font templates are local: no network, OCR service, or model download."""
    # 函数说明：读取数字识别参数，生成 0～9 的 OpenCV 字体模板，并加载可用的本地 TTF 字体模板。
    def __init__(self, config):
        self.config = config['measurement']
        self.options = config['opencv']
        self.templates = {digit: [] for digit in range(10)}
        for digit in range(10):
            for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX):
                for thickness in (2, 3, 5, 7, 9):
                    img = np.zeros((100, 100), np.uint8)
                    cv2.putText(img, str(digit), (10, 80), font, 2.4, 255, thickness, cv2.LINE_AA)
                    self.templates[digit].append(normalize_glyph(img))
        try:
            from PIL import Image, ImageDraw, ImageFont
            for path in self.config.get('digit_font_paths', []):
                if not Path(path).is_file():
                    continue
                try:
                    font = ImageFont.truetype(path, 80)
                    for digit in range(10):
                        img = Image.new('L', (120, 120))
                        ImageDraw.Draw(img).text((10, 0), str(digit), font=font, fill=255)
                        self.templates[digit].append(normalize_glyph(np.asarray(img)))
                except (OSError, ValueError):
                    continue
        except ImportError:
            pass
        templates, identifiers = [], []
        for digit, variants in self.templates.items():
            templates.extend(variants)
            identifiers.extend([digit]*len(variants))
        self.template_stack = np.stack(templates)
        self.template_digits = np.asarray(identifiers)
        self.template_area = np.count_nonzero(self.template_stack, axis=(1, 2))
        self.dilated_stack = np.stack([cv2.dilate(t, np.ones((3, 3), np.uint8)) for t in templates])

    # 函数说明：将输入字形与模板比较，返回编号和得分；得分不足或候选区分度不足时返回空编号。
    def recognize(self, glyph):
        glyph = normalize_glyph(glyph)
        area = max(1, np.count_nonzero(glyph))
        dilated = cv2.dilate(glyph, np.ones((3, 3), np.uint8))
        overlap = np.count_nonzero(self.template_stack & glyph, axis=(1, 2))
        precision = np.count_nonzero(self.dilated_stack & glyph, axis=(1, 2))/area
        recall = np.count_nonzero(self.template_stack & dilated, axis=(1, 2))/np.maximum(1, self.template_area)
        dice = 2*overlap/(area+self.template_area)
        template_scores = .5*dice+.25*(precision+recall)
        scores = [(float(np.max(template_scores[self.template_digits == digit])), digit) for digit in range(10)]
        scores.sort(reverse=True)
        score, digit = scores[0]
        if score < self.config['digit_min_score'] or score-scores[1][0] < self.config['digit_score_margin']:
            return None, score
        return digit, score

    # 函数说明：从原始目标掩膜提取封闭白色数字，并将可靠编号关联到包含该字形的最小正方形候选。
    def label(self, paper, shapes, mask=None):
        from .vision import binary
        foreground = binary(paper, self.options) if mask is None else mask
        geometry, _ = filled_geometry(foreground)
        white = cv2.bitwise_and(cv2.bitwise_not(foreground), geometry)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(white)
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if area < 8 or h < 6 or h > paper.shape[0]*.25 or w > h*1.3:
                continue
            owners = [s for s in shapes if cv2.pointPolygonTest(s.corners.astype(np.float32), tuple(centroids[i]), True) > 3]
            owners = [s for s in owners if .08*np.sqrt(cv2.contourArea(s.corners)) <= h
                      <= .95*np.sqrt(cv2.contourArea(s.corners))]
            if not owners:
                continue
            glyph = (labels[y:y+h, x:x+w] == i).astype(np.uint8)*255
            digit, score = self.recognize(glyph)
            if digit is not None:
                # A visible glyph belongs to the smallest containing square.
                owner = min(owners, key=lambda s: s.size_mm)
                if owner.digit is None or score > getattr(owner, '_digit_score', 0):
                    owner.digit, owner._digit_score = digit, score


# 函数说明：按容差合并邻近的一维坐标，返回每组均值，用于汇总同一条边的多次线段观测。
def cluster(values, tolerance=3):
    groups = []
    for value in sorted(values):
        if groups and value-np.mean(groups[-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.mean(g)) for g in groups]


# 函数说明：将采样坐标取整并限制在掩膜范围内，返回各位置是否属于黑色目标前景。
def sample(mask, points):
    p = np.rint(points).astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, mask.shape[1]-1)
    p[:, 1] = np.clip(p[:, 1], 0, mask.shape[0]-1)
    return mask[p[:, 1], p[:, 0]] > 0


# 函数说明：检查已填充数字孔洞的几何掩膜，返回候选正方形有效性和可见边得分；完整边候选允许更多遮挡。
def square_support(mask, corners, complete_side=False):
    # Occluded samples (black on both sides) are neutral. Four partial sides
    # can fix a square even when neither diagonal pair of corners is visible.
    support = []
    center = corners.mean(axis=0)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        points = a[None, :] + np.linspace(.04, .96, 80)[:, None]*(b-a)[None, :]
        inward = center-(a+b)/2
        inward = inward/np.linalg.norm(inward)*3
        support.append(float(np.mean(sample(mask, points+inward) & ~sample(mask, points-inward))))
    grid = np.linspace(.08, .92, 16)
    points = np.array([corners[0] + u*(corners[1]-corners[0]) + v*(corners[3]-corners[0]) for u in grid for v in grid])
    fill = float(np.mean(sample(mask, points)))
    valid = np.all(np.asarray(support) >= .12) and np.mean(support) >= .20
    if complete_side:
        valid = valid or (np.count_nonzero(np.asarray(support) >= .04) >= 2
                          and max(support) >= .8 and np.mean(support) >= .28)
    return valid and fill >= .92, float(np.mean(support)*fill)


class Advanced:
    # 函数说明：保存发挥题配置，并创建共享纸面处理器及离线数字识别器。
    def __init__(self, config):
        self.config = config
        self.processor = PaperProcessor(config)
        self.digits = DigitRecognizer(config)

    # 函数说明：联合外轮廓直检、局部垂直边重建和鲁棒拟合，返回去重后以毫米计的正方形候选。
    def find_squares(self, mask):
        scale = self.processor.scale
        lo, hi = np.asarray(self.config['measurement']['square_size_range_mm'])*scale
        # Fill white glyphs only for geometry; OCR still receives the original paper.
        mask, contours = filled_geometry(mask)
        candidates = [Shape('square', float(np.mean(np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)))/scale,
                            order_quad(q)) for q in contour_squares(contours, lo, hi)]
        boundary_points = np.concatenate(contours).reshape(-1, 2) if contours else np.empty((0, 2))
        # LSD sees external fragments without the distracting digit strokes.
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(mask)[0]
        if lines is None:
            return candidates
        raw_segments = lines.reshape(-1, 2, 2)
        lengths = np.linalg.norm(raw_segments[:, 1]-raw_segments[:, 0], axis=1)
        raw_segments = raw_segments[np.argsort(-lengths)[:256]]
        segments = raw_segments
        lengths = np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)
        segments = segments[lengths > max(12, lo*.10)][:80]
        orientations = orthogonal_directions(segments)
        # A complete visible side also fixes a square when its opposite side is
        # entirely hidden by another black square. Both endpoints must meet an
        # observed perpendicular edge, preventing a truncated side from fixing x.
        for segment in segments:
            a, b = segment
            side = np.linalg.norm(b-a)
            if not lo-5 <= side <= hi+5:
                continue
            direction = (b-a)/side
            normal = np.array([-direction[1], direction[0]])
            endpoints = []
            for endpoint in (a, b):
                junctions = []
                for other in raw_segments:
                    vector = other[1]-other[0]
                    length = np.linalg.norm(vector)
                    if length < 6 or abs(np.dot(vector/length, direction)) >= .07:
                        continue
                    t, fraction = np.linalg.solve(np.column_stack((direction, -vector)), other[0]-a)
                    junction = a+t*direction
                    distance = np.linalg.norm(junction-endpoint)
                    if distance < max(9, side*.15) and -.15 <= fraction <= 1.15:
                        junctions.append((distance, junction))
                if junctions:
                    endpoints.append(min(junctions, key=lambda item: item[0])[1])
            if len(endpoints) != 2:
                continue
            a, b = endpoints
            side = np.linalg.norm(b-a)
            if not lo-5 <= side <= hi+5:
                continue
            for sign in (-1, 1):
                corners = np.array([a, b, b+sign*side*normal, a+sign*side*normal], np.float32)
                if np.any(corners < 0) or np.any(corners[:, 0] >= mask.shape[1]) or np.any(corners[:, 1] >= mask.shape[0]):
                    continue
                if not (convex_corner(mask, corners[0], corners[1]-corners[0], corners[3]-corners[0])
                        and convex_corner(mask, corners[1], corners[0]-corners[1], corners[2]-corners[1])):
                    continue
                valid, score = square_support(mask, corners, complete_side=True)
                if valid:
                    candidate = Shape('square', side/scale, order_quad(corners), score)
                    candidate._complete_side = True
                    candidates.append(candidate)
        for angle in orientations[:12]:
            u = np.array([np.cos(angle), np.sin(angle)])
            v = np.array([-u[1], u[0]])
            xs, ys = [], []
            for segment in segments:
                vector = segment[1]-segment[0]
                vector /= np.linalg.norm(vector)
                if abs(vector @ v) > .998:
                    xs.append(segment.mean(axis=0) @ u)
                elif abs(vector @ u) > .998:
                    ys.append(segment.mean(axis=0) @ v)
            xs, ys = cluster(xs), cluster(ys)
            for x0, x1 in combinations(xs, 2):
                side = x1-x0
                if not lo-5 <= side <= hi+5:
                    continue
                for y0, y1 in combinations(ys, 2):
                    if abs(y1-y0-side) > max(4, side*.035):
                        continue
                    corners = np.array([x0*u+y0*v, x1*u+y0*v, x1*u+y1*v, x0*u+y1*v], np.float32)
                    if np.any(corners < 0) or np.any(corners[:, 0] >= mask.shape[1]) or np.any(corners[:, 1] >= mask.shape[0]):
                        continue
                    valid, score = square_support(mask, corners)
                    if valid:
                        candidates.append(Shape('square', (side+y1-y0)/2/scale, order_quad(corners), score))
        refined = []
        for shape in candidates:
            corners = refine_square(mask, shape.corners, boundary_points)
            valid, score = square_support(mask, corners, getattr(shape, '_complete_side', False))
            size = float(np.mean(np.linalg.norm(np.roll(corners, -1, axis=0)-corners, axis=1)))/scale
            if valid and lo/scale-2 <= size <= hi/scale+2:
                refined.append(Shape('square', size, corners, score))
        # Keep the best-supported fit for the same target, rather than retaining
        # slightly shortened copies inferred from a fragment endpoint.
        unique = []
        for s in sorted(refined, key=lambda s: -s.score):
            if not any(abs(s.size_mm-t.size_mm) < max(3., .055*min(s.size_mm, t.size_mm))
                       and np.linalg.norm(s.corners.mean(axis=0)-t.corners.mean(axis=0)) < 8 for t in unique):
                unique.append(s)
        return unique

    # 函数说明：测量单帧中的最小正方形或指定编号正方形，返回距离、边长及标注结果；识别不可靠时抛出异常。
    def measure(self, frame, digit=None):
        if digit is not None and digit not in range(10):
            raise ValueError('编号必须为 0～9')
        prepared = self.processor.prepare(frame)
        shapes = self.find_squares(prepared[2])
        if not shapes:
            raise ValueError('未找到可靠的正方形边界，请检查遮挡、边框和光照')
        if digit is not None:
            self.digits.label(prepared[1], shapes, prepared[2])
            matches = [s for s in shapes if s.digit == digit]
            if len(matches) != 1:
                raise ValueError(f'未能唯一识别编号 {digit}，请调整画面清晰度')
            selected = matches[0]
        else:
            selected = min(shapes, key=lambda s: s.size_mm)
        return self.processor.result(prepared, shapes, selected)
