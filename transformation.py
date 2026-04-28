import numpy as np
import cv2

def create_translation_matrix(tx, ty):
    return np.array([[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)], [0.0, 0.0, 1.0]])

def create_scale_matrix(s):
    return np.array([[float(s), 0.0, 0.0], [0.0, float(s), 0.0], [0.0, 0.0, 1.0]])

def extract_features(image, binary_mask):
    return cv2.bitwise_and(image, image, mask=binary_mask)

def apply_translation(image, binary_mask, tx, ty):
    h, w = image.shape[:2]
    T = create_translation_matrix(tx, ty)
    T_2x3 = T[:2, :]
    extracted_person = extract_features(image, binary_mask)
    translated_person = cv2.warpAffine(extracted_person, T_2x3, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    translated_mask = cv2.warpAffine(binary_mask, T_2x3, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (translated_person, translated_mask)

def apply_transformation(image, binary_mask, tx, ty, s, cx, cy):
    h, w = image.shape[:2]
    T_to_origin = create_translation_matrix(-cx, -cy)
    S = create_scale_matrix(s)
    T_back = create_translation_matrix(cx, cy)
    T_translate = create_translation_matrix(tx, ty)
    Total_T = T_translate @ T_back @ S @ T_to_origin
    T_2x3 = Total_T[:2, :]
    extracted_person = extract_features(image, binary_mask)
    transformed_person = cv2.warpAffine(extracted_person, T_2x3, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    transformed_mask = cv2.warpAffine(binary_mask, T_2x3, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (transformed_person, transformed_mask)
