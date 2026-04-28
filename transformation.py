import numpy as np


def create_translation_matrix(tx, ty):
    
    return np.array(
        [[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def create_scale_matrix(s):
    return np.array(
        [[float(s), 0.0, 0.0], [0.0, float(s), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def extract_features(image, binary_mask):
    mask_bool = binary_mask > 0
    if image.ndim == 3:
        return image * mask_bool[:, :, None]
    return image * mask_bool


def _warp_affine_nearest(src, transform_3x3, out_h, out_w, fill_value=0):
    
    inv_t = np.linalg.inv(transform_3x3)

    yy, xx = np.indices((out_h, out_w), dtype=np.float64)
    ones = np.ones_like(xx)
    dst_h = np.stack((xx, yy, ones), axis=0).reshape(3, -1)

    src_h = inv_t @ dst_h
    src_x = np.rint(src_h[0]).astype(np.int64)
    src_y = np.rint(src_h[1]).astype(np.int64)

    valid = (src_x >= 0) & (src_x < src.shape[1]) & (src_y >= 0) & (src_y < src.shape[0])

    if src.ndim == 3:
        out = np.full((out_h, out_w, src.shape[2]), fill_value, dtype=src.dtype)
        out_2d = out.reshape(-1, src.shape[2])
        out_2d[valid] = src[src_y[valid], src_x[valid]]
    else:
        out = np.full((out_h, out_w), fill_value, dtype=src.dtype)
        out_flat = out.reshape(-1)
        out_flat[valid] = src[src_y[valid], src_x[valid]]

    return out


def _ensure_binary_mask(mask):
    return (mask > 0).astype(mask.dtype)


def apply_translation(image, binary_mask, tx, ty):
    h, w = image.shape[:2]
    t = create_translation_matrix(tx, ty)
    extracted_person = extract_features(image, binary_mask)
    translated_person = _warp_affine_nearest(extracted_person, t, h, w, fill_value=0)
    translated_mask = _warp_affine_nearest(binary_mask, t, h, w, fill_value=0)
    translated_mask = _ensure_binary_mask(translated_mask)
    return (translated_person, translated_mask)


def apply_transformation(image, binary_mask, tx, ty, s, cx, cy):
    h, w = image.shape[:2]
    t_to_origin = create_translation_matrix(-cx, -cy)
    s_mat = create_scale_matrix(s)
    t_back = create_translation_matrix(cx, cy)
    t_translate = create_translation_matrix(tx, ty)
    total_t = t_translate @ t_back @ s_mat @ t_to_origin

    extracted_person = extract_features(image, binary_mask)
    transformed_person = _warp_affine_nearest(extracted_person, total_t, h, w, fill_value=0)
    transformed_mask = _warp_affine_nearest(binary_mask, total_t, h, w, fill_value=0)
    transformed_mask = _ensure_binary_mask(transformed_mask)
    return (transformed_person, transformed_mask)
