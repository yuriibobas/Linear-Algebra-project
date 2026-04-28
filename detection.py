import cv2
import numpy as np

_CUSTOM_WIN_W = 64
_CUSTOM_WIN_H = 128
_CUSTOM_CELL = 8
_CUSTOM_BINS = 9
_CUSTOM_BIN_WIDTH = 180.0 / _CUSTOM_BINS
_CUSTOM_BLOCK_EPS = 1e-6
_CUSTOM_SCORE_OFFSET = 10.0
_PEOPLE_SVM = None
_HOG_DETECTOR = None


def preprocess_for_hog(image):
    """Apply CLAHE on L-channel to normalise local contrast before HOG.

    This follows the gamma / colour normalisation recommendation from
    Dalal & Triggs (2005) and improves gradient stability under varying
    illumination.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def shrink_boxes(boxes, factor=0.1):
    """Shrink bounding boxes inward by *factor* to tighter-fit the figure.

    HOG returns enlarged boxes; trimming them typically improves IoU with
    ground-truth annotations.
    """
    result = []
    for (x, y, w, h) in boxes:
        dx = int(w * factor / 2)
        dy = int(h * factor / 2)
        result.append((x + dx, y + dy, w - 2 * dx, h - 2 * dy))
    return result


def _get_default_people_svm():
    """Load OpenCV's default linear SVM coefficients once."""
    global _PEOPLE_SVM
    if _PEOPLE_SVM is None:
        det = cv2.HOGDescriptor_getDefaultPeopleDetector().astype(np.float32)
        # OpenCV stores free coefficient as -rho for this detector vector.
        _PEOPLE_SVM = (det[:-1], float(-det[-1]))
    return _PEOPLE_SVM


def _get_opencv_hog_detector():
    """Build and cache OpenCV HOG person detector once."""
    global _HOG_DETECTOR
    if _HOG_DETECTOR is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _HOG_DETECTOR = hog
    return _HOG_DETECTOR


def _compute_hog_descriptor(gray_window):
    """Compute Dalal-Triggs style HOG for a 64x128 grayscale window."""
    gray = gray_window.astype(np.float32)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

    h, w = gray.shape
    cells_y = h // _CUSTOM_CELL
    cells_x = w // _CUSTOM_CELL

    if cells_y != 16 or cells_x != 8:
        return None

    hist = np.zeros((cells_y, cells_x, _CUSTOM_BINS), dtype=np.float32)

    flat_ang = ang.ravel()
    flat_mag = mag.ravel()
    yy, xx = np.indices((h, w))
    cell_y = (yy // _CUSTOM_CELL).ravel()
    cell_x = (xx // _CUSTOM_CELL).ravel()

    bin_pos = flat_ang / _CUSTOM_BIN_WIDTH
    bin_low = np.floor(bin_pos).astype(np.int32) % _CUSTOM_BINS
    bin_high = (bin_low + 1) % _CUSTOM_BINS
    w_high = bin_pos - np.floor(bin_pos)
    w_low = 1.0 - w_high

    np.add.at(hist, (cell_y, cell_x, bin_low), flat_mag * w_low)
    np.add.at(hist, (cell_y, cell_x, bin_high), flat_mag * w_high)

    blocks = []
    for by in range(cells_y - 1):
        for bx in range(cells_x - 1):
            block = hist[by:by + 2, bx:bx + 2, :].ravel()
            norm = np.linalg.norm(block) + _CUSTOM_BLOCK_EPS
            block = block / norm
            block = np.clip(block, 0.0, 0.2)
            block = block / (np.linalg.norm(block) + _CUSTOM_BLOCK_EPS)
            blocks.append(block)

    if not blocks:
        return None
    return np.concatenate(blocks).astype(np.float32)


def _nms_xywh(boxes, scores, iou_threshold):
    """Pure NumPy NMS for xywh boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-6)

        order = rest[iou <= iou_threshold]

    return keep


def _detect_humans_custom(image, win_stride, scale, hit_threshold):
    """Sliding-window multi-scale detector with custom HOG + linear SVM."""
    svm_w, svm_b = _get_default_people_svm()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boxes = []
    scores = []
    scale_factor = 1.0
    h0, w0 = gray.shape

    while True:
        sw = int(w0 / scale_factor)
        sh = int(h0 / scale_factor)
        if sw < _CUSTOM_WIN_W or sh < _CUSTOM_WIN_H:
            break

        scaled = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_LINEAR)
        max_y = sh - _CUSTOM_WIN_H + 1
        max_x = sw - _CUSTOM_WIN_W + 1

        for y in range(0, max_y, win_stride[1]):
            for x in range(0, max_x, win_stride[0]):
                window = scaled[y:y + _CUSTOM_WIN_H, x:x + _CUSTOM_WIN_W]
                feat = _compute_hog_descriptor(window)
                if feat is None:
                    continue
                raw_score = float(np.dot(svm_w, feat) + svm_b)
                # Align score scale with existing project thresholds (~0..1).
                score = raw_score - _CUSTOM_SCORE_OFFSET
                if score < hit_threshold:
                    continue

                x0 = int(round(x * scale_factor))
                y0 = int(round(y * scale_factor))
                ww = int(round(_CUSTOM_WIN_W * scale_factor))
                hh = int(round(_CUSTOM_WIN_H * scale_factor))
                boxes.append((x0, y0, ww, hh))
                scores.append(score)

        scale_factor *= scale

    return boxes, scores


def detect_humans(
    image,
    win_stride=(8, 8),
    padding=(8, 8),
    scale=1.05,
    score_threshold=0.0,
    nms_threshold=0.25,
    hit_threshold=0.0,
    use_preprocessing=True,
    shrink_factor=0.0,
    backend="opencv",
):
    """Detect humans using HOG + linear SVM.

    Parameters
    ----------
    hit_threshold : float
        SVM decision threshold passed directly to ``detectMultiScale``.
        Higher values → fewer but more confident raw detections.
    use_preprocessing : bool
        If True, apply CLAHE contrast normalisation before HOG extraction.
    shrink_factor : float
        Fraction by which to shrink each returned box (0 = no shrink).
    """
    if use_preprocessing:
        image = preprocess_for_hog(image)

    if backend == "custom":
        # Custom detector ignores explicit padding and runs pure sliding windows.
        boxes, weights = _detect_humans_custom(
            image=image,
            win_stride=win_stride,
            scale=scale,
            hit_threshold=hit_threshold,
        )
    else:
        hog = _get_opencv_hog_detector()
        boxes, weights = hog.detectMultiScale(
            image,
            winStride=win_stride,
            padding=padding,
            scale=scale,
            hitThreshold=hit_threshold,
        )

    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    keep = weights > score_threshold
    boxes = boxes[keep]
    weights = weights[keep]
    if len(boxes) == 0:
        return []

    keep_idx = _nms_xywh(boxes, weights, iou_threshold=nms_threshold)
    if len(keep_idx) == 0:
        return boxes.tolist()

    result = boxes[keep_idx].tolist()

    if shrink_factor > 0:
        result = shrink_boxes(result, factor=shrink_factor)

    return result


def get_binary_masks(
    image,
    boxes,
    top_k=2,
    grabcut_iters=5,
    morph_kernel_size=5,
    morph_open_iters=2,
    morph_close_iters=2,
):
    masks = []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:top_k]

    for (x, y, w, h) in boxes:
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w)
        h = min(image.shape[0] - y, h)
        rect = (x, y, w, h)

        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, grabcut_iters, cv2.GC_INIT_WITH_RECT)

        binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iters
        )
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iters
        )
        # Final erosion to trim GrabCut edge artifacts (background halo)
        erode_kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.erode(binary_mask, erode_kernel, iterations=2)

        # Keep only the connected component containing the box center.
        # This removes stray GrabCut patches (e.g., wall segments) that
        # are disconnected from the actual person silhouette.
        cx = x + w // 2
        cy = y + h // 2
        num_labels, labels = cv2.connectedComponents(binary_mask)
        if num_labels > 1 and cy < labels.shape[0] and cx < labels.shape[1]:
            center_label = labels[cy, cx]
            if center_label > 0:
                binary_mask = (labels == center_label).astype(np.uint8)
            else:
                # Center pixel is background — find the largest component
                best_label, best_count = 0, 0
                for lbl in range(1, num_labels):
                    count = int(np.sum(labels == lbl))
                    if count > best_count:
                        best_count = count
                        best_label = lbl
                binary_mask = (labels == best_label).astype(np.uint8)

        masks.append((binary_mask, rect))

    return masks


def filter_person_boxes(
    boxes,
    image_shape,
    min_area_ratio=0.01,
    max_area_ratio=0.65,
    min_aspect=0.2,
    max_aspect=1.2,
):
    """Drop unlikely person boxes using simple geometry priors.

    This helps remove tiny false positives and implausible regions
    before swap selection.
    """
    if boxes is None:
        return []

    h, w = image_shape[:2]
    image_area = float(max(1, h * w))
    filtered = []

    for b in boxes:
        x, y, bw, bh = [int(v) for v in b]
        if bw <= 0 or bh <= 0:
            continue
        area_ratio = (bw * bh) / image_area
        aspect = bw / float(max(1, bh))
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if aspect < min_aspect or aspect > max_aspect:
            continue
        filtered.append((x, y, bw, bh))

    return filtered


def pick_swap_pair(boxes, min_center_distance_ratio=0.12, image_shape=None):
    """Pick two boxes that are both large and sufficiently separated."""
    if boxes is None or len(boxes) < 2:
        return []

    boxes = list(boxes)
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    if image_shape is None:
        return boxes[:2]

    h, w = image_shape[:2]
    min_dist = min_center_distance_ratio * float(max(h, w))

    best_pair = None
    best_score = -1.0

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1 = boxes[i]
            b2 = boxes[j]
            c1x = b1[0] + b1[2] / 2.0
            c1y = b1[1] + b1[3] / 2.0
            c2x = b2[0] + b2[2] / 2.0
            c2y = b2[1] + b2[3] / 2.0
            dist = float(np.hypot(c2x - c1x, c2y - c1y))
            if dist < min_dist:
                continue
            area_score = float(b1[2] * b1[3] + b2[2] * b2[3])
            score = area_score + dist * 10.0
            if score > best_score:
                best_score = score
                best_pair = [b1, b2]

    if best_pair is not None:
        return best_pair

    return boxes[:2]
