# mnf_transform_improved_final.py

import numpy as np
import gc  # Garbage collector

# Memory-safe MSE calculation with chunking
def compute_mse_chunked(X_original, X_reconstructed, chunk_size=1000):
    """
    Compute MSE in chunks to avoid memory issues
    """
    mse = 0.0
    n_total = X_original.shape[0]
    
    for start_idx in range(0, n_total, chunk_size):
        end_idx = min(start_idx + chunk_size, n_total)
        diff = X_original[start_idx:end_idx] - X_reconstructed[start_idx:end_idx]
        mse += np.sum(diff ** 2)
        
        # Force garbage collection
        del diff
        gc.collect()
        
    mse = mse / (X_original.shape[0] * X_original.shape[1])
    return mse

# Memory-efficient covariance computation
def compute_covariance_chunked(X, chunk_size=1000):
    """
    Compute covariance matrix in chunks
    """
    n_samples, n_features = X.shape
    X_mean = np.mean(X, axis=0)
    cov = np.zeros((n_features, n_features), dtype=np.float64)
    
    # Calculate covariance in chunks
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        X_chunk = X[start_idx:end_idx] - X_mean
        cov += X_chunk.T @ X_chunk
        
        # Force garbage collection
        del X_chunk
        gc.collect()
        
    cov /= (n_samples - 1)
    return cov

# Manual eigen-decomposition using SVD
def compute_eigen_decomposition(C):
    """
    Compute eigenvalues and eigenvectors using SVD
    """
    U, S, Vt = np.linalg.svd(C)
    eigenvals = S
    eigenvecs = U
    return eigenvals, eigenvecs

# Find elbow point manually (no external libraries)
def find_elbow_point(x_values, y_values):
    """
    Find the elbow point in a curve using maximum curvature method.
    
    Args:
        x_values: Component numbers (e.g., [1, 2, 3, ...])
        y_values: Error metrics (e.g., MSE values)
        
    Returns:
        elbow_index: Index of the elbow point
    """
    # Normalize the x and y values to [0,1] range for proper scaling
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # Avoid division by zero
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1
    
    # Normalize coordinates
    x_norm = [(x - x_min) / x_range for x in x_values]
    y_norm = [(y - y_min) / y_range for y in y_values]
    
    # Calculate distances from each point to the line connecting first and last points
    # Line equation: Ax + By + C = 0
    first_point = (x_norm[0], y_norm[0])
    last_point = (x_norm[-1], y_norm[-1])
    
    # If first and last points are the same, return the first point
    if first_point == last_point:
        return 0
        
    # Calculate line parameters
    A = last_point[1] - first_point[1]
    B = first_point[0] - last_point[0]
    C = (last_point[0] * first_point[1]) - (first_point[0] * last_point[1])
    
    # Calculate distance from each point to line
    distances = []
    
    for i in range(len(x_norm)):
        numerator = abs(A * x_norm[i] + B * y_norm[i] + C)
        denominator = (A**2 + B**2)**0.5  # Square root
        
        # Avoid division by zero
        if denominator == 0:
            distances.append(0)
        else:
            distances.append(numerator / denominator)
    
    # Find index of maximum distance
    elbow_index = distances.index(max(distances))
    
    # Return the index in the original array
    return elbow_index

# Main MNF Transform Function with improved memory management
def mnf_transform_chunked(X, n_components_list, chunk_size=1000):
    """
    Perform MNF transform with memory optimization.
    
    Args:
        X: Hyperspectral image (M, N, K)
        n_components_list: List of number of components to keep
        chunk_size: Size of chunks for processing
        
    Returns:
        mse_list: List of MSE for each n
        reconstructed_list: List of reconstructed images
        mnf_components: MNF transformed data with all components
    """
    M, N, K = X.shape
    n_pixels = M * N
    X_reshaped = X.reshape(n_pixels, K)  # Reshape to (pixels, bands)
    
    print(f"Image reshaped to {X_reshaped.shape}")
    
    # Step 1: Estimate noise covariance matrix Σn using spatial differences
    print("Computing noise matrix...")
    noise = X[:, :-1, :] - X[:, 1:, :]
    noise_reshaped = noise.reshape(-1, K)
    Sigma_n = compute_covariance_chunked(noise_reshaped, chunk_size)
    
    # Free memory
    del noise, noise_reshaped
    gc.collect()
    
    # Step 2: Center the original data
    print("Centering data...")
    X_mean = np.mean(X_reshaped, axis=0)
    
    # Step 3: Eigen-decomposition of Σn
    print("Computing noise eigendecomposition...")
    eigenvals_n, eigenvecs_n = compute_eigen_decomposition(Sigma_n)
    
    # Free memory
    del Sigma_n
    gc.collect()
    
    # Step 4: Whitening matrix
    print("Computing whitening matrix...")
    # Handle very small eigenvalues to avoid numerical instability
    min_eigenval = np.max(eigenvals_n) * 1e-10
    eigenvals_n = np.maximum(eigenvals_n, min_eigenval)
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvals_n))
    F = Lambda_inv_sqrt @ eigenvecs_n.T
    
    # Free memory
    del Lambda_inv_sqrt
    gc.collect()
    
    # Step 5: Whiten the centered signal in chunks
    print("Whitening data in chunks...")
    Y = np.zeros((n_pixels, K), dtype=np.float64)
    
    for start_idx in range(0, n_pixels, chunk_size):
        end_idx = min(start_idx + chunk_size, n_pixels)
        X_chunk_centered = X_reshaped[start_idx:end_idx] - X_mean
        Y[start_idx:end_idx] = X_chunk_centered @ F.T
        
        # Free memory
        del X_chunk_centered
        gc.collect()
    
    # Step 6: Covariance matrix of whitened data
    print("Computing whitened data covariance...")
    Sigma_y = compute_covariance_chunked(Y, chunk_size)
    
    # Step 7: Eigen-decomposition of Σy
    print("Computing MNF components...")
    eigenvals_y, eigenvecs_y = compute_eigen_decomposition(Sigma_y)
    
    # Free memory
    del Sigma_y
    gc.collect()
    
    # Step 8: Project into MNF space in chunks
    print("Projecting to MNF space in chunks...")
    Z = np.zeros((n_pixels, K), dtype=np.float64)
    
    for start_idx in range(0, n_pixels, chunk_size):
        end_idx = min(start_idx + chunk_size, n_pixels)
        Z[start_idx:end_idx] = Y[start_idx:end_idx] @ eigenvecs_y
    
    # Store MNF components for visualization
    mnf_components = Z.reshape(M, N, K)
    
    # Free memory for Y as it's no longer needed
    del Y
    gc.collect()
    
    # Prepare for reconstructions
    mse_list = []
    reconstructed_list = []
    
    # Calculate F_inv once (reused for all reconstructions)
    F_inv = eigenvecs_n @ np.diag(np.sqrt(eigenvals_n))
    
    print("Computing reconstructions for different component counts...")
    for n in n_components_list:
        print(f"Processing {n} components...")
        
        # Step 9: Select first n MNF components
        V_n = eigenvecs_y[:, :n]
        
        # Step 10 & 11: Reconstruct in chunks
        X_approx = np.zeros((n_pixels, K), dtype=np.float64)
        
        for start_idx in range(0, n_pixels, chunk_size):
            end_idx = min(start_idx + chunk_size, n_pixels)
            # Reconstruct whitened data using inverse PCA
            Z_chunk = Z[start_idx:end_idx, :n]
            Y_approx = Z_chunk @ V_n.T
            
            # Inverse whitening
            X_approx[start_idx:end_idx] = Y_approx @ F_inv.T + X_mean
            
            # Free memory
            del Z_chunk, Y_approx
            gc.collect()
        
        # Step 12: Calculate MSE
        mse = compute_mse_chunked(X_reshaped, X_approx, chunk_size)
        mse_list.append(mse)
        
        # Reshape back to image dimensions
        reconstructed_list.append(X_approx.reshape(M, N, K))
        
        # Free memory
        del X_approx
        gc.collect()
    
    # Step 13: Return MSE list, reconstructed images, and MNF components
    return mse_list, reconstructed_list, mnf_components