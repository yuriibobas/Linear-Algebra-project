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
        raise ValueError("Images must have the same dimensions.")
        
    diff = np.subtract(imageA, imageB, dtype=np.float64)
    mse = np.mean(diff ** 2)
    return mse

def compute_classification_metrics(tp, fp, fn):
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1
