import numpy as np

def calculate_iou(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    x_left = max(xA, xB)
    y_top = max(yA, yB)
    x_right = min(xA + wA, xB + wB)
    y_bottom = min(yA + hA, yB + hB)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = wA * hA
    boxB_area = wB * hB
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

def calculate_mse(imageA, imageB):
    if imageA.shape != imageB.shape:
        raise ValueError('Images must have the same dimensions.')
    diff = np.subtract(imageA, imageB, dtype=np.float64)
    mse = np.mean(diff ** 2)
    return mse

def compute_classification_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return (precision, recall, f1)

def evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    if pred_boxes is None:
        pred_boxes = []
    if gt_boxes is None:
        gt_boxes = []
    pred_boxes = list(pred_boxes)
    gt_boxes = list(gt_boxes)
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mean_iou_matched': 0.0}
    matched_gt = set()
    matched_ious = []
    tp = 0
    for p in pred_boxes:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = calculate_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j != -1 and best_iou >= iou_threshold:
            matched_gt.add(best_j)
            matched_ious.append(best_iou)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision, recall, f1 = compute_classification_metrics(tp, fp, fn)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'mean_iou_matched': mean_iou}
