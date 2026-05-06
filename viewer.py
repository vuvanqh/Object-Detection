import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ObjectDetectionViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("DISASTER RESPONSE UAS: SIMPLE OBJECT DETECTION OPERATOR VIEWER")
        self.root.geometry("1200x600")
        self.root.configure(bg="#2b2b2b")
        
        self.model = None
        # Try loading a default model if available
        self.default_model_path = "best.pt" 
        if YOLO is not None and os.path.exists(self.default_model_path):
            try:
                self.model = YOLO(self.default_model_path)
            except Exception as e:
                print(f"Failed to load model: {e}")
        
        self.image_paths = []
        self.current_image_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#1a1a1a", height=40)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="DISASTER RESPONSE UAS: SIMPLE OBJECT DETECTION OPERATOR VIEWER", 
                                bg="#1a1a1a", fg="white", font=("Arial", 14, "bold"))
        header_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Main content
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel (Files)
        left_panel = tk.Frame(main_frame, bg="#333333", width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        files_label = tk.Label(left_panel, text="FILES", bg="#1a1a1a", fg="white", font=("Arial", 10, "bold"))
        files_label.pack(side=tk.TOP, fill=tk.X)
        
        self.listbox = tk.Listbox(left_panel, bg="white", fg="black", selectbackground="#cccccc", font=("Arial", 10))
        self.listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        load_folder_btn = tk.Button(left_panel, text="LOAD FOLDER", bg="#444444", fg="white", font=("Arial", 10, "bold"), 
                                    command=self.load_folder, relief=tk.FLAT)
        load_folder_btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        
        load_model_btn = tk.Button(left_panel, text="LOAD MODEL", bg="#444444", fg="white", font=("Arial", 10, "bold"), 
                                   command=self.load_model, relief=tk.FLAT)
        load_model_btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        
        run_btn = tk.Button(left_panel, text="RUN DETECTION", bg="#1a1a1a", fg="white", font=("Arial", 10, "bold"), 
                            command=self.run_detection, relief=tk.FLAT)
        run_btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Middle Panel (Raw Image)
        mid_panel = tk.Frame(main_frame, bg="#333333")
        mid_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        raw_label = tk.Label(mid_panel, text="RAW AERIAL IMAGE", bg="#1a1a1a", fg="white", font=("Arial", 10, "bold"))
        raw_label.pack(side=tk.TOP, fill=tk.X)
        
        self.raw_canvas = tk.Label(mid_panel, bg="#444444", text="No Image Selected", fg="white", font=("Arial", 12))
        self.raw_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right Panel (Predicted Image)
        right_panel = tk.Frame(main_frame, bg="#333333")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        pred_label = tk.Label(right_panel, text="PREDICTED IMAGE (HUMAN / TENT)", bg="#1a1a1a", fg="white", font=("Arial", 10, "bold"))
        pred_label.pack(side=tk.TOP, fill=tk.X)
        
        self.pred_canvas = tk.Label(right_panel, bg="#444444", text="No Prediction", fg="white", font=("Arial", 12))
        self.pred_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg="#1a1a1a", height=30)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        footer_frame.pack_propagate(False)
        
        authors_text = "Authors: Mateusz Blazejowski, Katarzyna Przech, Michal Szyc, Quoc Hoang Vu Van"
        footer_label = tk.Label(footer_frame, text=authors_text, bg="#1a1a1a", fg="#aaaaaa", font=("Arial", 14))
        footer_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Try loading default data folder
        default_data = os.path.join("data", "images", "test")
        if os.path.exists(default_data):
            self.load_images_from_folder(default_data)

    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            self.load_images_from_folder(folder_path)

    def load_images_from_folder(self, folder_path):
        self.image_paths = []
        self.listbox.delete(0, tk.END)
        valid_exts = {".jpg", ".jpeg", ".png"}
        for f in sorted(os.listdir(folder_path)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                full_path = os.path.join(folder_path, f)
                self.image_paths.append(full_path)
                self.listbox.insert(tk.END, f)

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select YOLO Model (.pt)", filetypes=[("PyTorch Model", "*.pt")])
        if model_path:
            if YOLO is None:
                messagebox.showerror("Error", "Ultralytics YOLO is not installed.")
                return
            try:
                self.model = YOLO(model_path)
                messagebox.showinfo("Success", f"Model loaded from {os.path.basename(model_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")

    def on_image_select(self, event):
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            self.current_image_path = self.image_paths[idx]
            self.display_image(self.current_image_path, self.raw_canvas)
            # Clear prediction canvas
            self.pred_canvas.configure(image='', text="Run Detection to see results")
            self.pred_canvas.image = None

    def display_image(self, path_or_img, label_widget):
        if isinstance(path_or_img, str):
            img = Image.open(path_or_img)
        else:
            img = path_or_img
            
        label_widget.update()
        canvas_width = label_widget.winfo_width()
        canvas_height = label_widget.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 400, 400
            
        # Maintain aspect ratio
        img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        label_widget.configure(image=photo, text="")
        label_widget.image = photo

    def run_detection(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
            
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a YOLO model first. Click 'LOAD MODEL'.")
            return
            
        try:
            # Change text to indicate processing
            self.pred_canvas.configure(image='', text="Processing...")
            self.root.update()
            
            # Run inference
            results = self.model(self.current_image_path)
            
            # Get the plotted image (BGR format from OpenCV)
            res_plotted = results[0].plot()
            
            # Convert BGR to RGB for PIL
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(res_rgb)
            
            self.display_image(pil_img, self.pred_canvas)
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
            self.pred_canvas.configure(text="Detection failed")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionViewer(root)
    root.mainloop()
