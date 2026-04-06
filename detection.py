import cv2
import numpy as np

def detect_humans(image):
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    boxes, weights = hog.detectMultiScale(image, winStride=(8,8), padding=(8,8), scale=1.05)
    return boxes

def get_binary_masks(image, boxes):
    
    masks = []
  
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)[:2]

    for (x, y, w, h) in boxes:
      
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w)
        h = min(image.shape[0] - y, h)
        rect = (x, y, w, h)

        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
       
        binary_mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        masks.append((binary_mask, rect))
        
    return masks
