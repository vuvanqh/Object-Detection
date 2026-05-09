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
        self.raw_original_image = None
        self.pred_original_image = None
        self.raw_zoom = 1.0
        self.pred_zoom = 1.0
        self.zoom_options = [25, 50, 75, 100, 125, 150, 175, 200, 300, 400]
        
        self.setup_ui()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#1a1a1a", height=40)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="DISASTER RESPONSE UAS: SIMPLE OBJECT DETECTION OPERATOR VIEWER", 
                                bg="#1a1a1a", fg="white", font=("Arial", 14, "bold"))
        header_label.pack(side=tk.LEFT, padx=10, pady=5)

        help_btn = tk.Button(
            header_frame,
            text="HELP",
            bg="#444444",
            fg="white",
            font=("Arial", 10, "bold"),
            relief=tk.FLAT,
            command=self.show_help,
        )
        help_btn.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Main content
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel (Files)
        left_panel = tk.Frame(main_frame, bg="#333333", width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        files_label = tk.Label(left_panel, text="FILES", bg="#1a1a1a", fg="white", font=("Arial", 10, "bold"))
        files_label.pack(side=tk.TOP, fill=tk.X)
        
        listbox_frame = tk.Frame(left_panel, bg="#333333")
        listbox_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        listbox_frame.grid_rowconfigure(0, weight=1)
        listbox_frame.grid_columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(listbox_frame, bg="white", fg="black", selectbackground="#cccccc", font=("Arial", 10))
        files_scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=files_scrollbar.set)

        self.listbox.grid(row=0, column=0, sticky="nsew")
        files_scrollbar.grid(row=0, column=1, sticky="ns")
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
        
        image_panes = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, bg="#2b2b2b", sashwidth=8, showhandle=False)
        image_panes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Middle Panel (Raw Image)
        mid_panel = tk.Frame(image_panes, bg="#333333")
        image_panes.add(mid_panel, minsize=220)

        raw_label = tk.Label(mid_panel, text="RAW AERIAL IMAGE", bg="#1a1a1a", fg="white", font=("Arial", 10, "bold"))
        raw_label.pack(side=tk.TOP, fill=tk.X)

        raw_controls = tk.Frame(mid_panel, bg="#333333")
        raw_controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 0))

        tk.Label(raw_controls, text="Zoom:", bg="#333333", fg="white", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.raw_zoom_var = tk.StringVar(value="100%")
        raw_zoom_menu = tk.OptionMenu(raw_controls, self.raw_zoom_var, *[f"{z}%" for z in self.zoom_options], command=lambda _: self.set_zoom("raw"))
        raw_zoom_menu.configure(bg="#444444", fg="white", highlightthickness=0, relief=tk.FLAT)
        raw_zoom_menu["menu"].configure(bg="white", fg="black")
        raw_zoom_menu.pack(side=tk.LEFT)

        raw_unzoom_btn = tk.Button(raw_controls, text="UNZOOM", bg="#444444", fg="white", font=("Arial", 9, "bold"),
                                   command=lambda: self.reset_zoom("raw"), relief=tk.FLAT)
        raw_unzoom_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.raw_canvas = tk.Canvas(mid_panel, bg="#444444", highlightthickness=0)
        self.raw_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.raw_canvas.bind("<Double-Button-1>", lambda _event: self.increment_zoom("raw"))
        self.raw_canvas.bind("<ButtonPress-1>", lambda event: self.start_pan("raw", event))
        self.raw_canvas.bind("<B1-Motion>", lambda event: self.move_pan("raw", event))
        self.raw_canvas.bind("<ButtonRelease-1>", lambda _event: self.end_pan("raw"))
        self.raw_canvas.bind("<Configure>", lambda _event: self.refresh_canvas("raw"))
        self.show_placeholder("raw", "No Image Selected")

        # Right Panel (Predicted Image)
        right_panel = tk.Frame(image_panes, bg="#333333")
        image_panes.add(right_panel, minsize=220)

        pred_label = tk.Label(right_panel, text="PREDICTED IMAGE (HUMAN / TENT)", bg="#1a1a1a", fg="white", font=("Arial", 10, "bold"))
        pred_label.pack(side=tk.TOP, fill=tk.X)

        pred_controls = tk.Frame(right_panel, bg="#333333")
        pred_controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 0))

        tk.Label(pred_controls, text="Zoom:", bg="#333333", fg="white", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.pred_zoom_var = tk.StringVar(value="100%")
        pred_zoom_menu = tk.OptionMenu(pred_controls, self.pred_zoom_var, *[f"{z}%" for z in self.zoom_options], command=lambda _: self.set_zoom("pred"))
        pred_zoom_menu.configure(bg="#444444", fg="white", highlightthickness=0, relief=tk.FLAT)
        pred_zoom_menu["menu"].configure(bg="white", fg="black")
        pred_zoom_menu.pack(side=tk.LEFT)

        pred_unzoom_btn = tk.Button(pred_controls, text="UNZOOM", bg="#444444", fg="white", font=("Arial", 9, "bold"),
                                    command=lambda: self.reset_zoom("pred"), relief=tk.FLAT)
        pred_unzoom_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.pred_canvas = tk.Canvas(right_panel, bg="#444444", highlightthickness=0)
        self.pred_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.pred_canvas.bind("<Double-Button-1>", lambda _event: self.increment_zoom("pred"))
        self.pred_canvas.bind("<ButtonPress-1>", lambda event: self.start_pan("pred", event))
        self.pred_canvas.bind("<B1-Motion>", lambda event: self.move_pan("pred", event))
        self.pred_canvas.bind("<ButtonRelease-1>", lambda _event: self.end_pan("pred"))
        self.pred_canvas.bind("<Configure>", lambda _event: self.refresh_canvas("pred"))
        self.show_placeholder("pred", "No Prediction")
        
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

    def show_help(self):
        instructions = (
            "How to use the viewer:\n\n"
            "1. Click LOAD FOLDER and select a directory with images.\n"
            "2. Select an image from the FILES list to show it in RAW AERIAL IMAGE.\n"
            "3. (Optional) Click LOAD MODEL to load a YOLO .pt model.\n"
            "4. Click RUN DETECTION to generate the predicted image.\n\n"
            "Image interaction:\n"
            "- Drag the middle bar between RAW and PREDICTED panels to resize them.\n"
            "- Double-click an image to zoom in by 25%.\n"
            "- Use each panel's Zoom dropdown to set exact zoom.\n"
            "- Click UNZOOM to reset zoom to 100% and recenter.\n"
            "- Click and drag on a zoomed image to pan around.\n"
        )
        messagebox.showinfo("Help - Viewer Instructions", instructions)

    def on_image_select(self, event):
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            self.current_image_path = self.image_paths[idx]
            self.raw_original_image = Image.open(self.current_image_path).convert("RGB")
            self.reset_zoom("raw")
            self.refresh_canvas("raw")
            # Clear prediction canvas
            self.pred_original_image = None
            self.reset_zoom("pred")
            self.show_placeholder("pred", "Run Detection to see results")

    def _get_state(self, target):
        if target == "raw":
            return self.raw_canvas, self.raw_original_image, self.raw_zoom
        return self.pred_canvas, self.pred_original_image, self.pred_zoom

    def _set_zoom_var(self, target):
        zoom_text = f"{int(round((self.raw_zoom if target == 'raw' else self.pred_zoom) * 100))}%"
        if target == "raw":
            self.raw_zoom_var.set(zoom_text)
        else:
            self.pred_zoom_var.set(zoom_text)

    def set_zoom(self, target):
        if target == "raw":
            selected = self.raw_zoom_var.get()
            self.raw_zoom = float(selected.rstrip("%")) / 100.0
        else:
            selected = self.pred_zoom_var.get()
            self.pred_zoom = float(selected.rstrip("%")) / 100.0
        self.refresh_canvas(target)

    def increment_zoom(self, target):
        if target == "raw":
            if self.raw_original_image is None:
                return
            self.raw_zoom *= 1.25
        else:
            if self.pred_original_image is None:
                return
            self.pred_zoom *= 1.25
        self._set_zoom_var(target)
        self.refresh_canvas(target)

    def reset_zoom(self, target):
        if target == "raw":
            self.raw_zoom = 1.0
        else:
            self.pred_zoom = 1.0
        self._set_zoom_var(target)
        self.refresh_canvas(target)
        canvas = self.raw_canvas if target == "raw" else self.pred_canvas
        canvas.xview_moveto(0.0)
        canvas.yview_moveto(0.0)

    def show_placeholder(self, target, text):
        canvas = self.raw_canvas if target == "raw" else self.pred_canvas
        canvas.delete("all")
        canvas.configure(scrollregion=(0, 0, max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)))
        canvas.image = None

    def start_pan(self, target, event):
        canvas, original_img, _ = self._get_state(target)
        if original_img is None:
            return
        canvas.scan_mark(event.x, event.y)

    def move_pan(self, target, event):
        canvas, original_img, _ = self._get_state(target)
        if original_img is None:
            return
        canvas.scan_dragto(event.x, event.y, gain=1)

    def end_pan(self, target):
        return

    def refresh_canvas(self, target):
        label_widget, original_img, zoom_factor = self._get_state(target)
        if original_img is None:
            return

        prev_x = label_widget.xview()[0]
        prev_y = label_widget.yview()[0]

        label_widget.update()
        canvas_width = max(label_widget.winfo_width(), 1)
        canvas_height = max(label_widget.winfo_height(), 1)
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 400, 400

        base_scale = min(canvas_width / original_img.width, canvas_height / original_img.height)
        render_scale = max(base_scale * zoom_factor, 0.01)
        render_width = max(int(original_img.width * render_scale), 1)
        render_height = max(int(original_img.height * render_scale), 1)

        rendered = original_img.resize((render_width, render_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(rendered)
        label_widget.delete("all")
        if render_width <= canvas_width and render_height <= canvas_height:
            x = (canvas_width - render_width) / 2
            y = (canvas_height - render_height) / 2
            label_widget.create_image(x, y, image=photo, anchor=tk.NW)
            label_widget.configure(scrollregion=(0, 0, canvas_width, canvas_height))
            label_widget.xview_moveto(0.0)
            label_widget.yview_moveto(0.0)
        else:
            label_widget.create_image(0, 0, image=photo, anchor=tk.NW)
            label_widget.configure(scrollregion=(0, 0, render_width, render_height))
            label_widget.xview_moveto(prev_x)
            label_widget.yview_moveto(prev_y)
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
            self.show_placeholder("pred", "Processing...")
            self.root.update()
            
            # Run inference
            results = self.model(self.current_image_path)
            
            # Get the plotted image (BGR format from OpenCV)
            res_plotted = results[0].plot()
            
            # Convert BGR to RGB for PIL
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            self.pred_original_image = Image.fromarray(res_rgb)
            self.reset_zoom("pred")
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
            self.show_placeholder("pred", "Detection failed")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionViewer(root)
    root.mainloop()
