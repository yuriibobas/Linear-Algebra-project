import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
import cv2

def _inpaint_single_scale(image, mask):
    h, w = image.shape[:2]
    missing_y, missing_x = np.where(mask > 0)
    num_missing = len(missing_y)
    if num_missing == 0:
        return image.copy()
    coord_to_idx = np.full((h, w), -1, dtype=np.int32)
    coord_to_idx[missing_y, missing_x] = np.arange(num_missing)
    row_idx = []
    col_idx = []
    values = []
    B = np.zeros((num_missing, 3), dtype=np.float64)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(num_missing):
        y = missing_y[i]
        x = missing_x[i]
        center_coef = 4.0
        for dy, dx in neighbors:
            ny, nx = (y + dy, x + dx)
            if ny < 0 or ny >= h or nx < 0 or (nx >= w):
                center_coef -= 1.0
                continue
            neighbor_idx = coord_to_idx[ny, nx]
            if neighbor_idx == -1:
                B[i] += image[ny, nx]
            else:
                row_idx.append(i)
                col_idx.append(neighbor_idx)
                values.append(-1.0)
        row_idx.append(i)
        col_idx.append(i)
        values.append(center_coef)
    A = coo_matrix((values, (row_idx, col_idx)), shape=(num_missing, num_missing)).tocsr()
    inpainted_image = image.copy().astype(np.float64)
    for c in range(3):
        X, info = cg(A, B[:, c], maxiter=max(2000, num_missing // 2))
        if info != 0:
            X = spsolve(A, B[:, c])
        inpainted_image[missing_y, missing_x, c] = np.clip(X, 0, 255)
    return inpainted_image.astype(np.uint8)

def inpaint_neighbor_averaging(image, mask):
    h, w = image.shape[:2]
    num_missing = np.sum(mask > 0)
    if num_missing == 0:
        return image.copy()
    if num_missing < 5000:
        return _inpaint_single_scale(image, mask)
    scale = 0.5
    small_h, small_w = (int(h * scale), int(w * scale))
    small_image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    coarse_result = _inpaint_single_scale(small_image, small_mask)
    upscaled = cv2.resize(coarse_result, (w, h), interpolation=cv2.INTER_LINEAR)
    init_image = image.copy()
    init_image[mask > 0] = upscaled[mask > 0]
    return _inpaint_single_scale(init_image, mask)
