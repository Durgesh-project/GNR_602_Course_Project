# Hyperspectral Image MNF Transform Tool

A GUI application for applying Minimum Noise Fraction (MNF) transform to hyperspectral images, with optimized memory management for large datasets.

## Overview

This tool implements the MNF (Minimum Noise Fraction) transform, a variation of Principal Component Analysis (PCA) specifically designed for hyperspectral image data. The MNF transform is particularly useful for:

- Noise reduction in hyperspectral data
- Dimensionality reduction
- Feature extraction
- Data compression

The application provides a user-friendly GUI that allows you to:
- Load hyperspectral images in various formats (.npy, .mat, .tif, .png, .jpg)
- Apply the MNF transform with memory-efficient processing
- Visualize the results through various plots and comparisons
- Automatically find the optimal number of components using the elbow method
- Save results for further analysis

## Features

- **Memory-Efficient Processing**: Uses chunking methods to handle large hyperspectral datasets
- **Interactive Visualization**: Compare original and reconstructed images with customizable RGB band selection
- **Automatic Component Selection**: Implements the elbow method to find optimal number of components
- **Multi-format Support**: Works with various file formats including NumPy arrays and MATLAB files
- **Center Crop Option**: Resize large images to focus on a specific region
- **Flexible Component Selection**: Choose which MNF components to use for reconstruction
- **Result Export**: Save MSE plots, reconstructed images, and MNF components

## Installation

### Prerequisites

- Python 3.6 or higher
- NumPy
- Matplotlib
- scikit-image
- SciPy
- tkinter

### Setup

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install numpy matplotlib scikit-image scipy
```

Note: tkinter usually comes pre-installed with Python. If not, you may need to install it separately based on your operating system.

## Usage

1. Run the main application:

```bash
python mnf_gui_improved_final.py
```

2. Load a hyperspectral image using the "Load Image" button
3. Configure your processing options:
   - Set the chunk size for memory management
   - Choose to crop the image if desired
   - Select which components to analyze
4. Click "Run MNF Transform" to process the image
5. Explore the results in the different tabs:
   - **MSE Plot**: View the Mean Squared Error vs. component count
   - **Results**: See detailed information about the optimal component count
   - **MNF Components**: Visualize individual MNF components
   - **Image Comparison**: Compare original vs. reconstructed images

## File Descriptions

- `mnf_gui_improved_final.py`: Main GUI application
- `mnf_transform_improved_final.py`: Core MNF transform algorithms and utility functions

## Technical Details

### MNF Transform Implementation

The MNF transform implemented in this software follows these key steps:

1. Estimate the noise covariance matrix using spatial differences
2. Whiten the data with respect to the noise covariance
3. Perform PCA on the whitened data
4. Project the data into MNF space
5. Reconstruct the data using selected components
6. Calculate reconstruction error (MSE)

### Memory Optimization

The implementation uses several techniques to minimize memory usage:

- Chunked processing for large matrices
- Garbage collection to free memory during processing
- Efficient covariance computation
- Manual eigen-decomposition using SVD

### Elbow Point Detection

The optimal number of components is found using the elbow method, which identifies the point where adding more components yields diminishing returns in terms of MSE reduction.

## Example Workflow

1. Load a hyperspectral image
2. Set chunk size to 1000 for a good balance of speed and memory usage
3. Optionally crop the image to focus on a region of interest
4. Select "all" components to analyze the full range or "1-20" to analyze only 20 components
5. Run the MNF transform
6. Check the MSE plot to see the elbow point (optimal component count)
7. Compare the original and reconstructed images
8. Save the results for further analysis

## License

This software is provided as-is for academic and research purposes.

## Credits

Developed for hyperspectral image analysis and processing.

## Troubleshooting

- **Memory Errors**: If you encounter memory issues, increase the chunk size or crop the image to a smaller size
- **File Loading Problems**: Ensure your file format is supported and properly structured
- **Performance Issues**: Close other applications when processing large images, or use a smaller crop size