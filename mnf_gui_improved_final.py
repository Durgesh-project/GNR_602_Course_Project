# mnf_gui_improved_final.py

import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage import io
from scipy import io as sio
import os
import gc  # Garbage collector
import threading

# Import MNF functions
from mnf_transform_improved_final import mnf_transform_chunked, find_elbow_point

class MNFGUI:
    def __init__(self, master):
        self.master = master
        master.title("MNF Transform GUI for Hyperspectral Images")
        master.geometry("900x800")
        
        # Main frame
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Image info
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Image Information")
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.image_info = ttk.Label(self.info_frame, text="No image loaded")
        self.image_info.pack(fill=tk.X, padx=5, pady=5)
        
        # Control panel
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.control_frame.pack(fill=tk.X, pady=15)

        # First row - Load button and chunk size
        self.load_button = ttk.Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=8, pady=10, sticky="w")

        ttk.Label(self.control_frame, text="Chunk Size:").grid(row=0, column=1, padx=5, pady=5)
        self.chunk_size_var = tk.StringVar(value="1000")
        self.chunk_size_entry = ttk.Entry(self.control_frame, textvariable=self.chunk_size_var, width=10)
        self.chunk_size_entry.grid(row=0, column=2, padx=5, pady=5)

        # Second row - Crop controls
        ttk.Label(self.control_frame, text="Crop to Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.crop_var = tk.BooleanVar(value=False)
        self.crop_check = ttk.Checkbutton(self.control_frame, variable=self.crop_var)
        self.crop_check.config(command=self.apply_crop)
        self.crop_check.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Width:").grid(row=1, column=2, padx=5, pady=5)
        self.crop_width_var = tk.StringVar(value="256")
        self.crop_width_entry = ttk.Entry(self.control_frame, textvariable=self.crop_width_var, width=5)
        self.crop_width_entry.bind("<Return>", lambda e: self.apply_crop())
        self.crop_width_entry.grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Height:").grid(row=1, column=4, padx=5, pady=5)
        self.crop_height_var = tk.StringVar(value="256")
        self.crop_height_entry = ttk.Entry(self.control_frame, textvariable=self.crop_height_var, width=5)
        self.crop_height_entry.bind("<Return>", lambda e: self.apply_crop())
        self.crop_height_entry.grid(row=1, column=5, padx=5, pady=5)

        # Third row - Components input
        ttk.Label(self.control_frame, text="Components:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.components_var = tk.StringVar(value="all")
        self.components_entry = ttk.Entry(self.control_frame, textvariable=self.components_var, width=10)
        self.components_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(self.control_frame, text="('all', '1,5,10' or '1-16')").grid(row=2, column=2, columnspan=3, padx=5, pady=5, sticky="w")

        # Help button for component input formats
        help_button = ttk.Button(self.control_frame, text="?", width=2, 
                                command=lambda: messagebox.showinfo("Component Format Help", 
                                "You can specify components in several ways:\n\n"
                                "- 'all': Use all available components\n"
                                "- Individual components: '1,5,10'\n"
                                "- Ranges: '1-16' (includes both 1 and 16)\n"
                                "- Combination: '1-5,8,10-12'\n\n"
                                "Components must be between 1 and the number of bands in the image."))
        help_button.grid(row=2, column=5, padx=(0, 5), pady=5)

        # Fourth row - Run button and progress bar
        self.run_button = ttk.Button(self.control_frame, text="Run MNF Transform", command=self.start_mnf)
        self.run_button.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, mode="indeterminate")
        self.progress.grid(row=3, column=1, columnspan=5, padx=5, pady=5, sticky="we")
        
        # Create tabs for visualization
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=15)
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        
        # MSE Plot tab
        self.mse_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.mse_frame, text="MSE Plot")
        
        # MNF Components tab
        self.components_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.components_frame, text="MNF Components")
        
        # Comparison tab
        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="Image Comparison")
        
        # Variable to store visualization data
        self.mse_list = None
        self.reconstructed_list = None
        self.mnf_components = None
        self.n_components_list = None
        
        # Create visualization plots
        self.create_mse_plot()
        
        # Add bands selection for hyperspectral visualization
        self.viz_frame = ttk.LabelFrame(self.comparison_frame, text="Bands for Visualization")
        self.viz_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.viz_frame, text="R:").grid(row=0, column=0, padx=5, pady=5)
        self.r_band = tk.StringVar(value="0")
        self.r_entry = ttk.Entry(self.viz_frame, textvariable=self.r_band, width=5)
        self.r_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.viz_frame, text="G:").grid(row=0, column=2, padx=5, pady=5)
        self.g_band = tk.StringVar(value="1")
        self.g_entry = ttk.Entry(self.viz_frame, textvariable=self.g_band, width=5)
        self.g_entry.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(self.viz_frame, text="B:").grid(row=0, column=4, padx=5, pady=5)
        self.b_band = tk.StringVar(value="2")
        self.b_entry = ttk.Entry(self.viz_frame, textvariable=self.b_band, width=5)
        self.b_entry.grid(row=0, column=5, padx=5, pady=5)
        
        self.update_viz_button = ttk.Button(self.viz_frame, text="Update Visualization", 
                                           command=self.update_visualization)
        self.update_viz_button.grid(row=0, column=6, padx=5, pady=5)
        
        # Component selection
        self.comp_frame = ttk.LabelFrame(self.comparison_frame, text="Components to Display")
        self.comp_frame.pack(fill=tk.X, pady=5)
        
        self.comp_var = tk.StringVar()
        self.comp_combobox = ttk.Combobox(self.comp_frame, textvariable=self.comp_var)
        self.comp_combobox.pack(padx=5, pady=5, side=tk.LEFT)
        
        # Save results button
        self.save_button = ttk.Button(self.comp_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(padx=5, pady=5, side=tk.RIGHT)
        
        # Place for comparison images
        self.img_frame = ttk.Frame(self.comparison_frame)
        self.img_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Numpy files", "*.npy"), 
                    ("MATLAB files", "*.mat"),
                    ("Tiff files", "*.tif *.tiff"), 
                    ("Image files", "*.png *.jpg *.jpeg")]
        )
        
        if not file_path:
            return
                
        try:
            self.status_var.set(f"Loading {os.path.basename(file_path)}...")
            
            # Load the image based on file type
            if file_path.endswith('.npy'):
                self.image = np.load(file_path)
            elif file_path.endswith('.mat'):
                # Load MATLAB file
                mat_contents = sio.loadmat(file_path)
                
                # Try to find the hyperspectral data
                # Common variable names in hyperspectral .mat files
                possible_var_names = ['data', 'hyperspectral', 'image', 'img', 'cube', 'hsi', 'hypercube']
                
                # Find the largest 3D array in the file
                largest_var = None
                largest_size = 0
                
                # First try with common names
                for var_name in possible_var_names:
                    if var_name in mat_contents and isinstance(mat_contents[var_name], np.ndarray):
                        array = mat_contents[var_name]
                        if array.ndim >= 3 and array.size > largest_size:
                            largest_var = array
                            largest_size = array.size
                
                # If no common name found, look for any 3D array
                if largest_var is None:
                    for var_name, array in mat_contents.items():
                        # Skip metadata variables (start with '__')
                        if var_name.startswith('__'):
                            continue
                        if isinstance(array, np.ndarray) and array.ndim >= 3 and array.size > largest_size:
                            largest_var = array
                            largest_size = array.size
                
                if largest_var is None:
                    raise ValueError("Could not find hyperspectral data in .mat file")
                
                # Make sure data is in the expected format [height, width, bands]
                # Some hyperspectral data is stored as [bands, height, width]
                if largest_var.ndim == 3:
                    # If bands is the first dimension (smaller than height/width), transpose
                    if largest_var.shape[0] < largest_var.shape[1] and largest_var.shape[0] < largest_var.shape[2]:
                        self.image = np.transpose(largest_var, (1, 2, 0))
                    else:
                        self.image = largest_var
                else:
                    # Handle higher dimensional arrays by reshaping
                    # Try to preserve the last dimension as bands
                    self.image = largest_var.reshape(-1, largest_var.shape[-2], largest_var.shape[-1])
            else:
                img = io.imread(file_path)
                if img.ndim == 2:  # If grayscale
                    img = img[:, :, np.newaxis]
                self.image = img
            
            # Store original image
            self.original_image = self.image.copy()
        
            # Apply cropping if selected
            if self.crop_var.get():
                try:
                    crop_width = int(self.crop_width_var.get())
                    crop_height = int(self.crop_height_var.get())
                    
                    # Check if crop dimensions are valid
                    if crop_width <= 0 or crop_height <= 0:
                        raise ValueError("Crop dimensions must be positive")
                    if crop_width > self.image.shape[1] or crop_height > self.image.shape[0]:
                        raise ValueError(f"Crop dimensions ({crop_width}x{crop_height}) exceed image size ({self.image.shape[1]}x{self.image.shape[0]})")
                    
                    # Calculate center crop
                    start_y = max(0, (self.image.shape[0] - crop_height) // 2)
                    end_y = start_y + crop_height
                    start_x = max(0, (self.image.shape[1] - crop_width) // 2)
                    end_x = start_x + crop_width
                    
                    # Apply crop
                    self.image = self.image[start_y:end_y, start_x:end_x, :]
                    self.status_var.set(f"Image loaded and cropped to {crop_width}x{crop_height}")
                except ValueError as e:
                    messagebox.showwarning("Crop Warning", str(e))
                
            # Update interface
            shape_text = f"Image loaded: {os.path.basename(file_path)}\n"
            shape_text += f"Shape: {self.image.shape}\n"
            shape_text += f"Dimensions: {self.image.shape[0]}×{self.image.shape[1]} pixels, {self.image.shape[2]} bands"
            self.image_info.config(text=shape_text)
            
            # Set default RGB bands based on total bands
            n_bands = self.image.shape[2]
            
            # For RGB images
            if n_bands == 3:
                self.r_band.set("0")
                self.g_band.set("1")
                self.b_band.set("2")
            # For hyperspectral images, suggest reasonable defaults
            elif n_bands > 10:
                self.r_band.set(str(n_bands // 4))
                self.g_band.set(str(n_bands // 2))
                self.b_band.set(str(3 * n_bands // 4))
                    
            # Show a preview
            self.display_image_preview()
            
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image")
            
    def apply_crop(self):
        """Apply cropping to the image when crop checkbox is toggled"""
        if not hasattr(self, 'original_image'):
            if hasattr(self, 'image'):
                # Store original image for first time
                self.original_image = self.image.copy()
            else:
                return  # No image loaded yet
        
        try:
            if self.crop_var.get():
                # Get crop dimensions
                crop_width = int(self.crop_width_var.get())
                crop_height = int(self.crop_height_var.get())
                
                # Validate crop dimensions
                if crop_width <= 0 or crop_height <= 0:
                    raise ValueError("Crop dimensions must be positive")
                if crop_width > self.original_image.shape[1] or crop_height > self.original_image.shape[0]:
                    raise ValueError(f"Crop dimensions ({crop_width}x{crop_height}) exceed image size ({self.original_image.shape[1]}x{self.original_image.shape[0]})")
                
                # Calculate center crop
                start_y = max(0, (self.original_image.shape[0] - crop_height) // 2)
                end_y = start_y + crop_height
                start_x = max(0, (self.original_image.shape[1] - crop_width) // 2)
                end_x = start_x + crop_width
                
                # Apply crop
                self.image = self.original_image[start_y:end_y, start_x:end_x, :].copy()
                self.status_var.set(f"Image cropped to {crop_width}x{crop_height}")
            else:
                # Restore original image
                self.image = self.original_image.copy()
                self.status_var.set("Original image restored (crop disabled)")
            
            # Update interface
            if hasattr(self, 'image_info'):
                shape_text = f"Shape: {self.image.shape}\n"
                shape_text += f"Dimensions: {self.image.shape[0]}×{self.image.shape[1]} pixels, {self.image.shape[2]} bands"
                self.image_info.config(text=shape_text)
            
            # Update preview
            self.display_image_preview()
            
        except ValueError as e:
            messagebox.showwarning("Crop Warning", str(e))
            # Reset crop checkbox if there was an error
            self.crop_var.set(False)

    def display_image_preview(self):
        """Display a preview of the loaded image"""
        if not hasattr(self, 'image'):
            return
            
        try:
            # Get the band indices
            r = int(self.r_band.get())
            g = int(self.g_band.get())
            b = int(self.b_band.get())
            
            # Create a 3-band color composite
            preview = np.zeros((self.image.shape[0], self.image.shape[1], 3))
            
            # Normalize each band to 0-1 range
            for i, band_idx in enumerate([r, g, b]):
                if 0 <= band_idx < self.image.shape[2]:
                    band = self.image[:, :, band_idx].astype(float)
                    # Normalize band
                    band_min = np.min(band)
                    band_max = np.max(band)
                    if band_max > band_min:
                        preview[:, :, i] = (band - band_min) / (band_max - band_min)
            
            # Create a figure in the comparison tab
            for widget in self.img_frame.winfo_children():
                widget.destroy()
                
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.imshow(preview)
            ax.set_title("Image Preview")
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=self.img_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create preview: {str(e)}")

    def start_mnf(self):
        """Start MNF processing in a separate thread"""
        if not hasattr(self, 'image'):
            messagebox.showerror("Error", "No image loaded!")
            return
            
        # Get chunk size
        try:
            chunk_size = int(self.chunk_size_var.get())
            if chunk_size <= 0:
                raise ValueError("Chunk size must be positive")
        except ValueError:
            messagebox.showerror("Error", "Invalid chunk size. Please enter a positive integer.")
            return
            
                # Get components
        components_text = self.components_var.get().strip()
        if components_text.lower() == 'all':
            max_bands = self.image.shape[2]
            n_components_list = list(range(1, max_bands + 1))
        else:
            try:
                # Process input that may include ranges like "1-16" and individual numbers like "1,5,10"
                components = []
                for part in components_text.split(','):
                    part = part.strip()
                    if '-' in part:
                        # Handle range format "start-end"
                        start, end = map(int, part.split('-'))
                        if start > end:
                            raise ValueError(f"Invalid range: {start}-{end}. Start must be less than or equal to end.")
                        components.extend(range(start, end + 1))  # Include the end value
                    else:
                        # Handle individual number
                        components.append(int(part))
                
                # Remove duplicates and sort
                n_components_list = sorted(list(set(components)))
                
                # Check if values are valid
                if any(c <= 0 or c > self.image.shape[2] for c in n_components_list):
                    raise ValueError("Component numbers must be between 1 and the number of bands")
                    
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid components: {str(e)}")
                return
                
        # Show running UI
        self.run_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status_var.set("Running MNF transform...")
        
        # Store for later
        self.n_components_list = n_components_list
        
        # Run in a separate thread
        thread = threading.Thread(target=self.run_mnf_thread, 
                                 args=(n_components_list, chunk_size))
        thread.daemon = True
        thread.start()

    def run_mnf_thread(self, n_components_list, chunk_size):
        """Run MNF transform in a separate thread"""
        try:
            # Run the MNF transform with chunking
            mse_list, reconstructed_list, mnf_components = mnf_transform_chunked(
                self.image, n_components_list, chunk_size=chunk_size)
            
            # Store results
            self.mse_list = mse_list
            self.reconstructed_list = reconstructed_list
            self.mnf_components = mnf_components
            
            # Update plots from the main thread
            self.master.after(100, self.update_plots)
            
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", f"MNF processing failed: {str(e)}"))
            self.master.after(0, self.reset_ui)

    def update_plots(self):
        """Update all plots and UI after MNF processing"""
        # Reset UI
        self.progress.stop()
        self.run_button.config(state=tk.NORMAL)
        
        # Update MSE plot
        self.update_mse_plot()
        
        # Update component visualization
        self.update_components_plot()
        
        # Update comparison view
        self.update_comparison_view()
        
        # Update the dropdown values 
        combobox_values = [f"{i} Components" for i in self.n_components_list]

        # Add the elbow point option if available
        if hasattr(self, 'elbow_point'):
            combobox_values.insert(0, "Elbow Point")
            
        self.comp_combobox['values'] = combobox_values
        if self.comp_combobox['values']:
            self.comp_combobox.current(0)
            
        # Make sure the selection is updated when the dropdown is changed
        self.comp_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_comparison_view())
                
        # Show results tab
        self.notebook.select(self.mse_frame)
        
        # Update status
        self.status_var.set("MNF transform completed")
        
        # Force garbage collection
        gc.collect()

    def create_mse_plot(self):
        """Create initial MSE plot"""
        for widget in self.mse_frame.winfo_children():
            widget.destroy()
            
        self.mse_fig = Figure(figsize=(6, 4))
        self.mse_ax = self.mse_fig.add_subplot(111)
        self.mse_ax.set_title("MSE vs Number of MNF Components")
        self.mse_ax.set_xlabel("Number of Components")
        self.mse_ax.set_ylabel("Mean Squared Error (MSE)")
        self.mse_ax.grid(True)
        
        self.mse_canvas = FigureCanvasTkAgg(self.mse_fig, master=self.mse_frame)
        self.mse_canvas.draw()
        self.mse_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_mse_plot(self):
        """Update MSE plot with new data and highlight the elbow point"""
        if not self.mse_list or not self.n_components_list:
            return
            
        self.mse_ax.clear()
        self.mse_ax.plot(self.n_components_list, self.mse_list, marker='o', linestyle='-')
        self.mse_ax.set_title("MSE vs Number of MNF Components")
        self.mse_ax.set_xlabel("Number of Components")
        self.mse_ax.set_ylabel("Mean Squared Error (MSE)")
        self.mse_ax.grid(True)
        
        # Find the elbow point
        try:
            elbow_point = find_elbow_point(self.n_components_list, self.mse_list)
            self.elbow_point = elbow_point  # Store for later use
            
            # Find the MSE value at the elbow point
            elbow_idx = self.n_components_list.index(elbow_point)
            elbow_mse = self.mse_list[elbow_idx]
            
            # Highlight the elbow point
            self.mse_ax.plot([elbow_point], [elbow_mse], 'ro', markersize=10, 
                        label=f'Elbow Point: {elbow_point} components')
            
            # Add a vertical line at the elbow point
            self.mse_ax.axvline(x=elbow_point, color='r', linestyle='--', alpha=0.5)
            
            self.mse_ax.legend()
        except Exception as e:
            print(f"Error finding elbow point: {str(e)}")
        
        # Add percentage labels
        if len(self.mse_list) > 0:
            base_mse = self.mse_list[0]
            for i, mse in enumerate(self.mse_list):
                if i % max(1, len(self.mse_list) // 10) == 0:  # Label only some points
                    pct_reduction = (1 - mse / base_mse) * 100
                    self.mse_ax.annotate(f"{pct_reduction:.1f}%", 
                                        xy=(self.n_components_list[i], mse),
                                        xytext=(5, 5), textcoords='offset points')
        
        self.mse_fig.tight_layout()
        self.mse_canvas.draw()
            
        # Also update the results text
        self.update_results_text()

    def update_results_text(self):
        """Update results text with MSE values and elbow point recommendation"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        results_text = tk.Text(self.results_frame, wrap=tk.WORD, height=20)
        results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(results_text, command=results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.config(yscrollcommand=scrollbar.set)
        
        # Add header
        results_text.insert(tk.END, "MNF Transform Results\n", "header")
        results_text.insert(tk.END, "---------------------\n\n")
        
        # Add elbow point recommendation
        if hasattr(self, 'elbow_point'):
            results_text.insert(tk.END, "Optimal Number of Components (Elbow Method):\n")
            results_text.insert(tk.END, f"Recommended: {self.elbow_point} components\n\n")
        
        # Add MSE results
        results_text.insert(tk.END, "Mean Squared Error by Components:\n")
        if self.mse_list and self.n_components_list:
            base_mse = self.mse_list[0]
            for i, n in enumerate(self.n_components_list):
                mse = self.mse_list[i]
                pct_reduction = (1 - mse / base_mse) * 100
                results_text.insert(tk.END, f"{n} Components: MSE = {mse:.6f} ")
                results_text.insert(tk.END, f"({pct_reduction:.2f}% reduction)\n")
                
                # Highlight the elbow point
                if hasattr(self, 'elbow_point') and n == self.elbow_point:
                    results_text.insert(tk.END, " --- OPTIMAL POINT (ELBOW METHOD) ---\n")
                    
        # Make it read-only
        results_text.config(state=tk.DISABLED)

    def update_components_plot(self):
        """Update MNF components visualization"""
        if self.mnf_components is None:
            return
            
        for widget in self.components_frame.winfo_children():
            widget.destroy()
            
        # Create controls for component selection
        control_frame = ttk.Frame(self.components_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Select component:").pack(side=tk.LEFT, padx=5)
        comp_var = tk.StringVar(value="1")
        
        comp_entry = ttk.Entry(control_frame, textvariable=comp_var, width=5)
        comp_entry.pack(side=tk.LEFT, padx=5)
        
        # Create figure
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        # Default component to display is the first one
        component_idx = 0
        component = self.mnf_components[:, :, component_idx]
        
        # Normalize for display
        vmin = np.min(component)
        vmax = np.max(component)
        
        # Display the component
        im = ax.imshow(component, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"MNF Component {component_idx+1}")
        ax.axis('off')
        
        # Add colorbar
        fig.colorbar(im, ax=ax)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=self.components_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Function to update component
        def update_component(event=None):
            try:
                component_idx = int(comp_var.get()) - 1
                if 0 <= component_idx < self.mnf_components.shape[2]:
                    component = self.mnf_components[:, :, component_idx]
                    
                    # Normalize for display
                    vmin = np.min(component)
                    vmax = np.max(component)
                    
                    # Update the image
                    im.set_data(component)
                    im.set_clim(vmin, vmax)
                    ax.set_title(f"MNF Component {component_idx+1}")
                    canvas.draw()
            except ValueError:
                pass
        
        # Connect update function to the entry
        comp_entry.bind("<Return>", update_component)

    def update_comparison_view(self):
        """Update comparison view with original and reconstructed images"""
        if not self.reconstructed_list or not self.n_components_list:
            return
            
        # Get the selected component count
        try:
            comp_text = self.comp_var.get()
            if not comp_text:
                # Default to elbow point if available, otherwise use first component
                if hasattr(self, 'elbow_point'):
                    comp_idx = self.n_components_list.index(self.elbow_point)
                else:
                    comp_idx = 0
            elif comp_text == "Elbow Point" and hasattr(self, 'elbow_point'):
                # Find the index of the elbow point in n_components_list
                comp_idx = self.n_components_list.index(self.elbow_point)
            else:
                # Extract the number of components from the text (e.g., "16 Components" -> 16)
                selected_comp = int(comp_text.split()[0])
                # Find the corresponding index in n_components_list
                comp_idx = self.n_components_list.index(selected_comp)
        except (ValueError, IndexError) as e:
            print(f"Error selecting component: {str(e)}")
            # Default to elbow point if available, otherwise use first component
            if hasattr(self, 'elbow_point'):
                comp_idx = self.n_components_list.index(self.elbow_point)
            else:
                comp_idx = 0
            
        # Get the reconstructed image
        reconstructed_img = self.reconstructed_list[comp_idx]
        n_components = self.n_components_list[comp_idx]
        
        # Update the comparison view
        self.show_comparison(reconstructed_img, n_components)

    def update_visualization(self):
        """Update visualization with selected RGB bands"""
        if not hasattr(self, 'image'):
            return
            
        # If we have reconstruction, show comparison
        if hasattr(self, 'reconstructed_list') and self.reconstructed_list:
            self.update_comparison_view()
        else:
            # Just show the original image preview
            self.display_image_preview()

    def show_comparison(self, reconstructed_img, n_components):
        """Show comparison between original and reconstructed images"""
        # Clear the frame
        for widget in self.img_frame.winfo_children():
            widget.destroy()
            
        # Get the band indices for visualization
        try:
            r = int(self.r_band.get())
            g = int(self.g_band.get())
            b = int(self.b_band.get())
            
            # Check if bands are in range
            if not (0 <= r < self.image.shape[2] and 
                   0 <= g < self.image.shape[2] and 
                   0 <= b < self.image.shape[2]):
                messagebox.showerror("Error", "Band indices out of range")
                return
                
            # Create the figure
            fig = Figure(figsize=(10, 7))
            
            # Original image
            ax1 = fig.add_subplot(121)
            original_rgb = self.create_rgb_composite(self.image, r, g, b)
            ax1.imshow(original_rgb)
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            # Reconstructed image
            ax2 = fig.add_subplot(122)
            reconstructed_rgb = self.create_rgb_composite(reconstructed_img, r, g, b)
            ax2.imshow(reconstructed_rgb)
            ax2.set_title(f"Reconstructed ({n_components} components)")
            ax2.axis('off')
            
            # Add the figure to the frame
            canvas = FigureCanvasTkAgg(fig, master=self.img_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Update when component selection changes
            self.comp_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_comparison_view())
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show comparison: {str(e)}")

    def create_rgb_composite(self, image, r_band, g_band, b_band):
        """Create RGB composite from selected bands"""
        rgb = np.zeros((image.shape[0], image.shape[1], 3))
        
        # Extract and normalize each band
        for i, band_idx in enumerate([r_band, g_band, b_band]):
            band = image[:, :, band_idx].astype(float)
            # Normalize to 0-1 range
            band_min = np.min(band)
            band_max = np.max(band)
            if band_max > band_min:
                rgb[:, :, i] = (band - band_min) / (band_max - band_min)
                
        return rgb

    def save_results(self):
        """Save results to files"""
        if not self.mse_list or not self.reconstructed_list:
            messagebox.showerror("Error", "No results to save")
            return
            
        # Ask for directory to save
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            return
            
        try:
            # Save MSE data
            mse_file = os.path.join(save_dir, "mnf_mse_results.txt")
            with open(mse_file, 'w') as f:
                f.write("Components,MSE,Reduction(%)\n")
                base_mse = self.mse_list[0]
                for i, n in enumerate(self.n_components_list):
                    mse = self.mse_list[i]
                    pct_reduction = (1 - mse / base_mse) * 100
                    f.write(f"{n},{mse:.6f},{pct_reduction:.2f}\n")
                    
            # Save MSE plot
            mse_plot_file = os.path.join(save_dir, "mnf_mse_plot.png")
            self.mse_fig.savefig(mse_plot_file, dpi=300, bbox_inches='tight')
            
            # Save current visualization
            viz_file = os.path.join(save_dir, "mnf_visualization.png")
            for widget in self.img_frame.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg):
                    widget.figure.savefig(viz_file, dpi=300, bbox_inches='tight')
                    break
                    
            # Save all MNF components
            mnf_file = os.path.join(save_dir, "mnf_components.npy")
            np.save(mnf_file, self.mnf_components)
            
            # Save the currently selected reconstructed image
            try:
                # Get the selected component count from the dropdown
                comp_text = self.comp_var.get()
                
                if comp_text == "Elbow Point" and hasattr(self, 'elbow_point'):
                    # Find the index of the elbow point in n_components_list
                    comp_idx = self.n_components_list.index(self.elbow_point)
                    n_components = self.elbow_point
                elif comp_text:
                    # Extract the number of components from the text (e.g., "16 Components" -> 16)
                    selected_comp = int(comp_text.split()[0])
                    # Find the corresponding index in n_components_list
                    comp_idx = self.n_components_list.index(selected_comp)
                    n_components = selected_comp
                else:
                    # Default to elbow point if available, otherwise use first component
                    if hasattr(self, 'elbow_point'):
                        comp_idx = self.n_components_list.index(self.elbow_point)
                        n_components = self.elbow_point
                    else:
                        comp_idx = 0
                        n_components = self.n_components_list[0]
                    
                # Save the currently selected reconstructed image
                recon_img = self.reconstructed_list[comp_idx]
                recon_file = os.path.join(save_dir, f"reconstructed_{n_components}_components.npy")
                np.save(recon_file, recon_img)
                
                # Removed the code that always saves the elbow point reconstruction

            except (ValueError, IndexError) as e:
                print(f"Error saving selected reconstruction: {str(e)}")
                # If there's an error, try to save at least the first reconstruction
                recon_file = os.path.join(save_dir, f"reconstructed_{self.n_components_list[0]}_components.npy")
                np.save(recon_file, self.reconstructed_list[0])
                
            messagebox.showinfo("Success", f"Results saved to {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MNFGUI(root)
    root.mainloop()