import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from detection import detect_humans, get_binary_masks
from transformation import apply_transformation
from inpainting import inpaint_neighbor_averaging

class GeometricApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Linear Algebra - Image Composition")
        
        self.I_orig = None
        self.bg_holes = None       
        self.bg_mask = None       
        
        
        self.sprites = []         
        self.active_sprite_idx = -1
        self.drag_data = {"x": 0, "y": 0}
        self.transformed_masks = []
        self.current_composite = None
        
      
        self.setup_ui()
        
    def setup_ui(self):
        
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(control_frame, text="1. Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(control_frame, text="2. Detect & Extract", command=self.detect).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(control_frame, text="3. Render & Inpaint", command=self.finalize_inpaint).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Please load an image.")
        tk.Label(control_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=15)
        
        
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
       
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
       
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
            
        self.I_orig = cv2.imread(path)
        if self.I_orig is None:
            messagebox.showerror("Error", "Could not load image.")
            return
            
       
        h, w = self.I_orig.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            self.I_orig = cv2.resize(self.I_orig, (int(w*scale), int(h*scale)))
            
        self.bg_holes = None
        self.sprites = []
        self.transformed_masks = []
        self.status_var.set("Image loaded. Press 'Detect & Extract'.")
        self.render_frame()
        self.root.geometry(f"{self.I_orig.shape[1] + 20}x{self.I_orig.shape[0] + 60}")
        
    def detect(self):
        if self.I_orig is None:
            return
            
        self.status_var.set("Detecting humans using HOG+SVM...")
        self.root.update()
        
        boxes = detect_humans(self.I_orig)
        if not len(boxes):
            messagebox.showinfo("Result", "No humans detected.")
            self.status_var.set("Ready.")
            return
            
        masks_data = get_binary_masks(self.I_orig, boxes)
        
        self.sprites = []
        combined_original_masks = np.zeros(self.I_orig.shape[:2], dtype=np.uint8)
        
        for (mask, rect) in masks_data:
            cx = rect[0] + rect[2] // 2
            cy = rect[1] + rect[3] // 2
            self.sprites.append({
                'orig_mask': mask,
                'cx': cx, 'cy': cy,
                'tx': 0, 'ty': 0, 's': 1.0
            })
            combined_original_masks = cv2.bitwise_or(combined_original_masks, mask)
            
        self.bg_mask = combined_original_masks
        self.bg_holes = self.I_orig.copy()
        self.bg_holes[combined_original_masks > 0] = [0, 0, 0]
        
        self.status_var.set(f"Extracted {len(self.sprites)} people. Drag them or use mousewheel to scale.")
        self.render_frame()
        
    def render_frame(self):
        if self.I_orig is None:
            return
            
        if self.bg_holes is None:
            display_img = self.I_orig.copy()
        else:
            display_img = self.bg_holes.copy()
            self.transformed_masks = []
            
            
            for sp in self.sprites:
                trans_img, trans_mask = apply_transformation(self.I_orig, sp['orig_mask'], sp['tx'], sp['ty'], sp['s'], sp['cx'], sp['cy'])
                self.transformed_masks.append(trans_mask)
                
               
                target_idx = trans_mask > 0
                display_img[target_idx] = trans_img[target_idx]
                
        self.current_composite = display_img
        
     
        display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(display_rgb)
        self.photo = ImageTk.PhotoImage(image=im)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
    def get_clicked_sprite(self, x, y):
        h, w = self.I_orig.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return -1
            
       
        for i in range(len(self.transformed_masks)-1, -1, -1):
            if self.transformed_masks[i][y, x] > 0:
                return i
        return -1
        
    def on_press(self, event):
        if not self.sprites:
            return
        idx = self.get_clicked_sprite(event.x, event.y)
        if idx != -1:
            self.active_sprite_idx = idx
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            
    def on_drag(self, event):
        if self.active_sprite_idx == -1:
            return
            
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        
        self.sprites[self.active_sprite_idx]['tx'] += dx
        self.sprites[self.active_sprite_idx]['ty'] += dy
        
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        
        self.render_frame()
        
    def on_release(self, event):
        self.active_sprite_idx = -1
        
    def on_scroll(self, event):
        if not self.sprites:
            return
            
        idx = self.get_clicked_sprite(event.x, event.y)
        if idx != -1:
           
            if event.delta > 0:
                self.sprites[idx]['s'] *= 1.05
            else:
                self.sprites[idx]['s'] /= 1.05
            self.render_frame()

    def finalize_inpaint(self):
        if not self.sprites:
            return
            
        self.status_var.set("Calculating Background Equation (Inpainting)... Please wait.")
        self.root.update()
        
       
        final_holes = self.bg_mask.copy()
        for t_mask in self.transformed_masks:
            final_holes[t_mask > 0] = 0
            
        if np.sum(final_holes > 0) == 0:
            self.status_var.set("No pixels need inpainting.")
            return
            
        final_img = inpaint_neighbor_averaging(self.current_composite, final_holes)
        cv2.imwrite("gui_result.jpg", final_img)
        
       
        display_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(display_rgb)
        self.photo = ImageTk.PhotoImage(image=im)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.bg_holes = None
        self.sprites = [] 
        self.status_var.set("Finished! Saved as gui_result.jpg")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = GeometricApp(root)
    root.mainloop()
