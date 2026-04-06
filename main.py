import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from detection import detect_humans, get_binary_masks
from transformation import apply_translation
from inpainting import inpaint_neighbor_averaging

def process_image(image_path, output_path="result_image.jpg"):
   
    I = cv2.imread(image_path)
    if I is None:
        print(f"Error: Could not load image {image_path}")
        return
    
   
    print("Detecting humans using HOG+SVM...")
    boxes = detect_humans(I)
    
    if len(boxes) < 2:
        print("Need at least 2 humans in the image to swap them.")
        return
        
    print(f"Detected {len(boxes)} humans. Using the two largest.")
    
    
    masks_data = get_binary_masks(I, boxes)
    
    mask_A, rect_A = masks_data[0] 
    mask_B, rect_B = masks_data[1]
    
    
    cA_x = rect_A[0] + rect_A[2] // 2
    cA_y = rect_A[1] + rect_A[3] // 2
    
   
    cB_x = rect_B[0] + rect_B[2] // 2
    cB_y = rect_B[1] + rect_B[3] // 2
    
    tA_x, tA_y = cB_x - cA_x, cB_y - cA_y
    tB_x, tB_y = cA_x - cB_x, cA_y - cB_y
    
    print(f"Swapping Figure A (center {cA_x},{cA_y}) and Figure B (center {cB_x},{cB_y})")
    
   
    translated_person_A, translated_mask_A = apply_translation(I, mask_A, tA_x, tA_y)
    translated_person_B, translated_mask_B = apply_translation(I, mask_B, tB_x, tB_y)
    
    Inew = I.copy()
    
    
    combined_original_masks = cv2.bitwise_or(mask_A, mask_B)
    Inew[combined_original_masks > 0] = [0, 0, 0] 
    
  
    target_A_idx = translated_mask_A > 0
    Inew[target_A_idx] = translated_person_A[target_A_idx]
    
    target_B_idx = translated_mask_B > 0
    Inew[target_B_idx] = translated_person_B[target_B_idx]
    
    
    final_holes = combined_original_masks.copy()
    final_holes[translated_mask_A > 0] = 0
    final_holes[translated_mask_B > 0] = 0
    
   
    cv2.imwrite("pre_inpainting_" + os.path.basename(output_path), Inew)
    
   
    print(f"Inpainting {np.sum(final_holes > 0)} missing pixels...")
   
    final_image = inpaint_neighbor_averaging(Inew, final_holes)
    
    cv2.imwrite(output_path, final_image)
    print(f"Saved successful result to {output_path}")
    
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("After Transformation")
    plt.imshow(cv2.cvtColor(Inew, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Final Inpainted")
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("plot_" + os.path.basename(output_path))
    print("Saved plot visualization.")

if __name__ == "__main__":
    if not os.path.exists("test_image.jpg"):
        print("Please place an image named 'test_image.jpg' in the directory with at least two separated people.")
        print("You can run 'python download_test.py' to get a sample!")
    else:
        process_image("test_image.jpg", "result_image.jpg")
