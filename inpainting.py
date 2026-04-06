import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import cv2

def inpaint_neighbor_averaging(image, mask):
   
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
            ny, nx = y + dy, x + dx
            
           
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
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
       
        X = spsolve(A, B[:, c])
        inpainted_image[missing_y, missing_x, c] = np.clip(X, 0, 255)
        
    return inpainted_image.astype(np.uint8)
