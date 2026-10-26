"""Geometry-only masks and robust orthogonal fitting for the advanced task."""
import cv2
import numpy as np


# 函数说明：仅在发挥题几何检测中填充封闭白色孔洞，并返回外轮廓；原始灰度图和 OCR 字形不变。
def filled_geometry(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = mask.copy()
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        region = np.zeros((height, width), np.uint8)
        shifted = contour-np.array([[[x, y]]], dtype=contour.dtype)
        cv2.drawContours(region, [shifted], -1, 255, cv2.FILLED)
        ink = np.count_nonzero((mask[y:y+height, x:x+width] > 0) & (region > 0))
        # A white digit may occupy much of a small square; a mostly empty frame
        # is not a solid target and must not become one by hole filling.
        if ink/max(1, np.count_nonzero(region)) >= .40:
            filled[y:y+height, x:x+width][region > 0] = 255
    return filled, contours


# 函数说明：按线段长度对模 90 度方向聚类，返回加权平均方向，避免用首条短线的偏角代表整组边。
def orthogonal_directions(segments, tolerance_deg=3):
    groups = []
    tolerance = np.deg2rad(tolerance_deg)
    for segment in segments:
        vector = segment[1]-segment[0]
        length = np.linalg.norm(vector)
        if length < 1:
            continue
        angle = np.arctan2(vector[1], vector[0]) % (np.pi/2)
        best = None
        for i, group in enumerate(groups):
            mean = np.arctan2(group[1], group[0])/4 % (np.pi/2)
            difference = abs(angle-mean)
            difference = min(difference, np.pi/2-difference)
            if difference < tolerance and (best is None or difference < best[0]):
                best = difference, i
        item = np.array([length*np.cos(4*angle), length*np.sin(4*angle), length])
        if best is None:
            groups.append(item)
        else:
            groups[best[1]] += item
    return [float(np.arctan2(g[1], g[0])/4 % (np.pi/2))
            for g in sorted(groups, key=lambda g: -g[2])[:12]]


# 函数说明：采样顶点四个象限，判断它是否是可见凸直角，防止把重叠轮廓的凹角当作完整边端点。
def convex_corner(mask, point, direction1, direction2):
    d1 = direction1/np.linalg.norm(direction1)
    d2 = direction2/np.linalg.norm(direction2)
    grid = np.array([(x, y) for x in (3., 4.5, 6.) for y in (3., 4.5, 6.)])
    occupancy = []
    for sign1, sign2 in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        points = point+sign1*grid[:, :1]*d1+sign2*grid[:, 1:]*d2
        pixels = np.rint(points).astype(int)
        if (np.any(pixels < 0) or np.any(pixels[:, 0] >= mask.shape[1])
                or np.any(pixels[:, 1] >= mask.shape[0])):
            return False
        occupancy.append(np.mean(mask[pixels[:, 1], pixels[:, 0]] > 0))
    return occupancy[0] >= .75 and max(occupancy[1:]) <= .35


# 函数说明：由外轮廓中的凸四边形直接提出独立正方形，避免编号内部笔画挤占线段候选。
def contour_squares(contours, minimum, maximum):
    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) < minimum**2*.75:
            continue
        poly = cv2.approxPolyDP(contour, .018*cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        vectors = np.roll(poly, -1, axis=0)-poly
        lengths = np.linalg.norm(vectors, axis=1)
        if np.min(lengths) < 1 or max(lengths)/min(lengths) > 1.08:
            continue
        cosine = np.abs(np.sum(vectors*np.roll(vectors, 1, axis=0), axis=1)/(lengths*np.roll(lengths, 1)))
        if max(cosine) > .12 or not minimum-5 <= np.mean(lengths) <= maximum+5:
            continue
        candidates.append(cv2.boxPoints(cv2.minAreaRect(contour)))
    return candidates


# 函数说明：使用三条以上可见边的轮廓点进行 Huber 加权正方形拟合，联合优化中心、边长和方向。
def refine_square(mask, corners, boundary_points):
    corners = np.asarray(corners, float)
    center = corners.mean(axis=0)
    side = float(np.mean(np.linalg.norm(np.roll(corners, -1, axis=0)-corners, axis=1)))
    angle = float(np.arctan2(*(corners[1]-corners[0])[::-1]))
    half = side/2
    normals = ((corners+np.roll(corners, -1, axis=0))/2-center)/half
    points, sides = [], []
    for i, normal in enumerate(normals):
        delta = boundary_points-center
        along = delta @ np.array([-normal[1], normal[0]])
        distance = delta @ normal-half
        selected = boundary_points[(np.abs(distance) <= 3.) & (np.abs(along) <= half*.92)]
        if len(selected) < 6:
            continue
        inside = np.rint(selected-2.5*normal).astype(int)
        outside = np.rint(selected+2.5*normal).astype(int)
        inside[:, 0] = np.clip(inside[:, 0], 0, mask.shape[1]-1)
        inside[:, 1] = np.clip(inside[:, 1], 0, mask.shape[0]-1)
        outside[:, 0] = np.clip(outside[:, 0], 0, mask.shape[1]-1)
        outside[:, 1] = np.clip(outside[:, 1], 0, mask.shape[0]-1)
        visible = (mask[inside[:, 1], inside[:, 0]] > 0) & (mask[outside[:, 1], outside[:, 0]] == 0)
        selected = selected[visible]
        if len(selected) >= 6:
            # Bound fitting cost on large contours while retaining each visible side.
            selected = selected[::max(1, len(selected)//100)]
            points.append(selected)
            sides.extend([i]*len(selected))
    if len(points) < 3:
        return corners.astype(np.float32)
    points, sides = np.concatenate(points), np.asarray(sides)
    initial_center, initial_angle = center.copy(), angle
    initial_half = half
    for _ in range(4):
        theta = angle-np.pi/2+sides*np.pi/2
        normal = np.column_stack((np.cos(theta), np.sin(theta)))
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        delta = points-center
        residual = np.sum(delta*normal, axis=1)-half
        jacobian = np.column_stack((-normal, -np.ones(len(points)), np.sum(delta*tangent, axis=1)))
        weights = np.sqrt(np.minimum(1., 1.2/np.maximum(np.abs(residual), 1e-6)))
        step, _, rank, _ = np.linalg.lstsq(jacobian*weights[:, None], -residual*weights, rcond=None)
        if rank < 4:
            return corners.astype(np.float32)
        center += step[:2]
        half += step[2]
        angle += step[3]
        if np.linalg.norm(center-initial_center) > 5 or abs(half-initial_half) > 4 or abs(angle-initial_angle) > np.deg2rad(3):
            return corners.astype(np.float32)
        if np.linalg.norm(step) < 1e-3:
            break
    u, v = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
    return np.float32([center-half*u-half*v, center+half*u-half*v,
                       center+half*u+half*v, center-half*u+half*v])
