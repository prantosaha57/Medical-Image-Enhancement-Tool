import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, filters, morphology, segmentation, measure, restoration
from skimage.util import img_as_ubyte, img_as_float, random_noise
from skimage.color import rgb2gray
from scipy import ndimage
import cv2
from PIL import Image
import io as bytesio

def main():
    st.set_page_config(layout="wide", page_title="Medical Image Enhancement Tool")
    
    st.title("Medical Image Enhancement Tool")
    st.sidebar.header("Controls")
    
    # Upload image
    uploaded_file = st.sidebar.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png", "tif", "dicom"])
    
    if uploaded_file is not None:
        # Convert the uploaded file to an image
        image = load_image(uploaded_file)
        
        # Convert to grayscale if it's a color image
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray_image = rgb2gray(image)
            image_type = "RGB (Converted to Grayscale)"
        else:
            gray_image = image
            image_type = "Grayscale"
        
        # Display original image information
        st.sidebar.write(f"Image Type: {image_type}")
        st.sidebar.write(f"Dimensions: {gray_image.shape}")
        st.sidebar.write(f"Intensity Range: [{gray_image.min():.3f}, {gray_image.max():.3f}]")
        
        # Main processing options
        processing_category = st.sidebar.selectbox(
            "Select Enhancement Category",
            ["Basic Enhancement", "Histogram Operations", "Noise Reduction", 
             "Edge Detection", "Segmentation", "Advanced Filters"]
        )
        
        # Create columns for original and processed images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(gray_image, clamp=True, use_column_width=True)
            
            # Original histogram
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(gray_image.ravel(), bins=256, range=(0, 1), density=True)
            ax.set_title("Histogram")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        
        # Process image based on selected category
        processed_image = None
        
        if processing_category == "Basic Enhancement":
            processed_image = process_basic_enhancement(gray_image)
        elif processing_category == "Histogram Operations":
            processed_image = process_histogram_operations(gray_image)
        elif processing_category == "Noise Reduction":
            processed_image = process_noise_reduction(gray_image)
        elif processing_category == "Edge Detection":
            processed_image = process_edge_detection(gray_image)
        elif processing_category == "Segmentation":
            processed_image = process_segmentation(gray_image)
        elif processing_category == "Advanced Filters":
            processed_image = process_advanced_filters(gray_image)
        
        # Display processed image
        if processed_image is not None:
            with col2:
                st.subheader("Processed Image")
                st.image(processed_image, clamp=True, use_column_width=True)
                
                # Processed histogram
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(processed_image.ravel(), bins=256, range=(0, 1), density=True)
                ax.set_title("Histogram")
                ax.set_xlabel("Pixel Intensity")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
            
            # Download button for processed image
            buffer = bytesio.BytesIO()
            pil_img = Image.fromarray((processed_image * 255).astype(np.uint8))
            pil_img.save(buffer, format="PNG")
            st.download_button(
                label="Download Processed Image",
                data=buffer.getvalue(),
                file_name="processed_medical_image.png",
                mime="image/png"
            )
            
            # Display difference image
            st.subheader("Difference Analysis")
            col3, col4 = st.columns(2)
            
            with col3:
                difference = np.abs(processed_image - gray_image)
                st.image(difference, clamp=True, caption="Difference Image", use_column_width=True)
                
            with col4:
                # Metrics
                st.write("Image Metrics:")
                st.write(f"Mean Absolute Difference: {np.mean(difference):.5f}")
                st.write(f"Max Difference: {np.max(difference):.5f}")
                st.write(f"PSNR: {calculate_psnr(gray_image, processed_image):.2f} dB")
                st.write(f"SSIM: {calculate_ssim(gray_image, processed_image):.5f}")

def load_image(uploaded_file):
    """Load an image from an uploaded file"""
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
    
    # Convert to float [0, 1]
    if img is not None:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_as_float(img)
    return None

def process_basic_enhancement(image):
    """Process basic image enhancement operations"""
    enhancement_type = st.sidebar.selectbox(
        "Enhancement Type",
        ["Contrast Adjustment", "Brightness Adjustment", "Gamma Correction", 
         "Logarithmic Transform", "Power-Law Transform"]
    )
    
    if enhancement_type == "Contrast Adjustment":
        alpha = st.sidebar.slider("Contrast Factor", 0.1, 5.0, 1.0, 0.1)
        return np.clip((image - 0.5) * alpha + 0.5, 0, 1)
    
    elif enhancement_type == "Brightness Adjustment":
        beta = st.sidebar.slider("Brightness Adjustment", -0.5, 0.5, 0.0, 0.01)
        return np.clip(image + beta, 0, 1)
    
    elif enhancement_type == "Gamma Correction":
        gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0, 0.1)
        return np.power(image, gamma)
    
    elif enhancement_type == "Logarithmic Transform":
        c = st.sidebar.slider("Scaling Factor", 0.1, 5.0, 1.0, 0.1)
        # Add small constant to avoid log(0)
        return c * np.log1p(image) / np.log1p(1)
    
    elif enhancement_type == "Power-Law Transform":
        gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0, 0.1)
        c = st.sidebar.slider("Scaling Factor", 0.1, 5.0, 1.0, 0.1)
        return c * np.power(image, gamma)

def process_histogram_operations(image):
    """Process histogram-based operations"""
    hist_op = st.sidebar.selectbox(
        "Histogram Operation",
        ["Histogram Equalization", "Adaptive Histogram Equalization (CLAHE)", 
         "Histogram Matching", "Contrast Stretching"]
    )
    
    if hist_op == "Histogram Equalization":
        return exposure.equalize_hist(image)
    
    elif hist_op == "Adaptive Histogram Equalization (CLAHE)":
        kernel_size = st.sidebar.slider("Kernel Size", 8, 100, 32, 8)
        clip_limit = st.sidebar.slider("Clip Limit", 0.01, 0.5, 0.03, 0.01)
        return exposure.equalize_adapthist(image, kernel_size=kernel_size, clip_limit=clip_limit)
    
    elif hist_op == "Histogram Matching":
        hist_mapping = st.sidebar.selectbox(
            "Target Distribution",
            ["Linear", "Gaussian", "Rayleigh", "Exponential"]
        )
        
        # Create reference histogram based on selection
        x = np.linspace(0, 1, 256)
        if hist_mapping == "Linear":
            reference = x
        elif hist_mapping == "Gaussian":
            mu = st.sidebar.slider("Mean", 0.0, 1.0, 0.5, 0.05)
            sigma = st.sidebar.slider("Standard Deviation", 0.05, 0.5, 0.1, 0.05)
            reference = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            reference = (reference - reference.min()) / (reference.max() - reference.min())
        elif hist_mapping == "Rayleigh":
            scale = st.sidebar.slider("Scale", 0.05, 0.5, 0.1, 0.05)
            reference = x * np.exp(-(x ** 2) / (2 * scale ** 2))
            reference = (reference - reference.min()) / (reference.max() - reference.min())
        elif hist_mapping == "Exponential":
            scale = st.sidebar.slider("Scale", 0.1, 10.0, 3.0, 0.1)
            reference = np.exp(-scale * x)
            reference = (reference - reference.min()) / (reference.max() - reference.min())
        
        # Create a reference image with the desired histogram
        ref_image = reference.reshape(1, -1)
        return exposure.match_histograms(image, ref_image)
    
    elif hist_op == "Contrast Stretching":
        p_low = st.sidebar.slider("Lower Percentile", 0.0, 20.0, 2.0, 1.0)
        p_high = st.sidebar.slider("Upper Percentile", 80.0, 100.0, 98.0, 1.0)
        return exposure.rescale_intensity(
            image, 
            in_range=tuple(np.percentile(image, (p_low, p_high))),
            out_range=(0, 1)
        )

def process_noise_reduction(image):
    """Process noise reduction operations"""
    # Add noise option for demonstration
    add_noise = st.sidebar.checkbox("Add Noise for Demonstration", False)
    if add_noise:
        noise_type = st.sidebar.selectbox(
            "Noise Type",
            ["Gaussian", "Salt & Pepper", "Poisson", "Speckle"]
        )
        noise_amount = st.sidebar.slider("Noise Amount", 0.01, 0.5, 0.05, 0.01)
        
        if noise_type == "Gaussian":
            noisy_image = random_noise(image, mode='gaussian', var=noise_amount)
        elif noise_type == "Salt & Pepper":
            noisy_image = random_noise(image, mode='s&p', amount=noise_amount)
        elif noise_type == "Poisson":
            noisy_image = random_noise(image, mode='poisson')
        else:  # Speckle
            noisy_image = random_noise(image, mode='speckle', var=noise_amount)
    else:
        noisy_image = image

    # Display noisy image if noise was added
    if add_noise:
        st.sidebar.image(noisy_image, caption="Noisy Image", width=150)
    
    # Filter selection
    filter_type = st.sidebar.selectbox(
        "Filter Type",
        ["Mean Filter", "Gaussian Filter", "Median Filter", "Bilateral Filter",
         "Non-Local Means", "Total Variation", "Wiener Filter", "Anisotropic Diffusion"]
    )
    
    if filter_type == "Mean Filter":
        kernel_size = st.sidebar.slider("Kernel Size", 3, 21, 5, 2)
        return ndimage.uniform_filter(noisy_image, size=kernel_size)
    
    elif filter_type == "Gaussian Filter":
        sigma = st.sidebar.slider("Sigma", 0.1, 10.0, 1.0, 0.1)
        return filters.gaussian(noisy_image, sigma=sigma)
    
    elif filter_type == "Median Filter":
        kernel_size = st.sidebar.slider("Kernel Size", 3, 21, 3, 2)
        return filters.median(noisy_image, np.ones((kernel_size, kernel_size)))
    
    elif filter_type == "Bilateral Filter":
        sigma_spatial = st.sidebar.slider("Spatial Sigma", 1, 20, 3, 1)
        sigma_range = st.sidebar.slider("Range Sigma", 0.01, 0.5, 0.1, 0.01)
        return restoration.denoise_bilateral(noisy_image, sigma_color=sigma_range, sigma_spatial=sigma_spatial)
    
    elif filter_type == "Non-Local Means":
        patch_size = st.sidebar.slider("Patch Size", 3, 9, 5, 2)
        patch_distance = st.sidebar.slider("Patch Distance", 1, 15, 7, 2)
        h = st.sidebar.slider("h (Filter Strength)", 0.01, 1.0, 0.1, 0.01)
        fast_mode = st.sidebar.checkbox("Fast Mode", True)
        return restoration.denoise_nl_means(noisy_image, patch_size=patch_size, patch_distance=patch_distance, 
                                          h=h, fast_mode=fast_mode)
    
    elif filter_type == "Total Variation":
        weight = st.sidebar.slider("Weight", 0.01, 1.0, 0.1, 0.01)
        return restoration.denoise_tv_chambolle(noisy_image, weight=weight)
    
    elif filter_type == "Wiener Filter":
        psf = np.ones((5, 5)) / 25  # Simple point spread function
        balance = st.sidebar.slider("Noise-Balance Parameter", 0.01, 10.0, 0.1, 0.01)
        return restoration.wiener(noisy_image, psf, balance)
    
    elif filter_type == "Anisotropic Diffusion":
        iterations = st.sidebar.slider("Iterations", 1, 50, 10, 1)
        kappa = st.sidebar.slider("Kappa (Edge Stopping)", 0.01, 100.0, 30.0, 1.0)
        gamma = st.sidebar.slider("Gamma (Step Size)", 0.01, 0.25, 0.15, 0.01)
        return filters.anisotropic_diffusion(noisy_image, kappa=kappa, gamma=gamma, num_iter=iterations)

def process_edge_detection(image):
    """Process edge detection operations"""
    edge_detector = st.sidebar.selectbox(
        "Edge Detector",
        ["Sobel", "Prewitt", "Scharr", "Roberts", "Canny", "Laplacian of Gaussian (LoG)", "Difference of Gaussians (DoG)"]
    )
    
    if edge_detector == "Sobel":
        return filters.sobel(image)
    
    elif edge_detector == "Prewitt":
        return filters.prewitt(image)
    
    elif edge_detector == "Scharr":
        return filters.scharr(image)
    
    elif edge_detector == "Roberts":
        return filters.roberts(image)
    
    elif edge_detector == "Canny":
        sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
        low_threshold = st.sidebar.slider("Low Threshold", 0.01, 0.99, 0.1, 0.01)
        high_threshold = st.sidebar.slider("High Threshold", low_threshold, 0.99, 0.2, 0.01)
        return feature.canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    
    elif edge_detector == "Laplacian of Gaussian (LoG)":
        sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
        return filters.laplace(filters.gaussian(image, sigma=sigma))
    
    elif edge_detector == "Difference of Gaussians (DoG)":
        sigma1 = st.sidebar.slider("Sigma 1", 0.1, 5.0, 1.0, 0.1)
        sigma2 = st.sidebar.slider("Sigma 2", sigma1, 10.0, sigma1 * 1.6, 0.1)
        return filters.gaussian(image, sigma=sigma1) - filters.gaussian(image, sigma=sigma2)

def process_segmentation(image):
    """Process image segmentation operations"""
    seg_method = st.sidebar.selectbox(
        "Segmentation Method",
        ["Thresholding", "Adaptive Thresholding", "K-Means", "Watershed", "Region Growing", "Chan-Vese"]
    )
    
    if seg_method == "Thresholding":
        threshold_type = st.sidebar.selectbox(
            "Threshold Type",
            ["Manual", "Otsu", "Yen", "Li", "Triangle", "Isodata"]
        )
        
        if threshold_type == "Manual":
            thresh_value = st.sidebar.slider("Threshold Value", 0.0, 1.0, 0.5, 0.01)
            segmented = image > thresh_value
        elif threshold_type == "Otsu":
            thresh_value = filters.threshold_otsu(image)
            segmented = image > thresh_value
        elif threshold_type == "Yen":
            thresh_value = filters.threshold_yen(image)
            segmented = image > thresh_value
        elif threshold_type == "Li":
            thresh_value = filters.threshold_li(image)
            segmented = image > thresh_value
        elif threshold_type == "Triangle":
            thresh_value = filters.threshold_triangle(image)
            segmented = image > thresh_value
        else:  # Isodata
            thresh_value = filters.threshold_isodata(image)
            segmented = image > thresh_value
            
        st.sidebar.write(f"Threshold value: {thresh_value:.4f}")
        return segmented.astype(float)
    
    elif seg_method == "Adaptive Thresholding":
        block_size = st.sidebar.slider("Block Size", 3, 99, 11, 2)
        constant = st.sidebar.slider("Constant", -0.5, 0.5, 0.0, 0.01)
        return filters.threshold_local(image, block_size=block_size, offset=constant) < image
    
    elif seg_method == "K-Means":
        st.warning("This is a simplified K-means for demonstration. For larger images, it may be slow.")
        k = st.sidebar.slider("Number of Clusters", 2, 10, 3, 1)
        # Reshape for k-means
        pixel_values = image.reshape(-1, 1)
        # Use sklearn's KMeans for simplicity
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=0).fit(pixel_values)
        # Reshape back to image dimensions
        segmented = kmeans.labels_.reshape(image.shape)
        # Normalize to [0,1]
        return segmented / (k - 1)
    
    elif seg_method == "Watershed":
        # Compute gradient for watershed
        gradient = filters.sobel(image)
        
        # Define markers
        markers_threshold = st.sidebar.slider("Markers Threshold", 0.0, 1.0, 0.3, 0.01)
        markers = np.zeros_like(image, dtype=np.int32)
        markers[image < markers_threshold] = 1
        markers[image > 1 - markers_threshold] = 2
        
        # Watershed segmentation
        segmented = segmentation.watershed(gradient, markers)
        return (segmented - 1) / 1.0  # Normalize to [0,1]
    
    elif seg_method == "Region Growing":
        # Simple region growing for demonstration
        seed_x = st.sidebar.slider("Seed X (% of width)", 0, 100, 50, 1) / 100
        seed_y = st.sidebar.slider("Seed Y (% of height)", 0, 100, 50, 1) / 100
        threshold = st.sidebar.slider("Threshold", 0.01, 0.5, 0.1, 0.01)
        
        # Convert percentage to pixel coordinates
        seed = (int(image.shape[0] * seed_y), int(image.shape[1] * seed_x))
        
        # Create mask and add seed point
        mask = np.zeros_like(image, dtype=bool)
        mask[seed] = True
        
        # Grow region
        segmented = segmentation.flood(image, seed, tolerance=threshold)
        return segmented.astype(float)
    
    elif seg_method == "Chan-Vese":
        iterations = st.sidebar.slider("Iterations", 10, 200, 50, 10)
        lambda1 = st.sidebar.slider("Lambda1", 0.1, 2.0, 1.0, 0.1)
        lambda2 = st.sidebar.slider("Lambda2", 0.1, 2.0, 1.0, 0.1)
        
        segmented = segmentation.chan_vese(image, mu=0.25, lambda1=lambda1, lambda2=lambda2, 
                                        tol=1e-3, max_iter=iterations, dt=0.5)
        return segmented.astype(float)

def process_advanced_filters(image):
    """Process advanced filtering operations"""
    filter_type = st.sidebar.selectbox(
        "Advanced Filter",
        ["Unsharp Masking", "Frangi Filter (Vessel Enhancement)", "Gabor Filter", 
         "Local Binary Pattern", "Gaussian Derivative", "Homomorphic Filtering"]
    )
    
    if filter_type == "Unsharp Masking":
        radius = st.sidebar.slider("Blur Radius", 0.1, 20.0, 5.0, 0.1)
        amount = st.sidebar.slider("Amount", 0.1, 5.0, 2.0, 0.1)
        
        # Gaussian blur
        blurred = filters.gaussian(image, sigma=radius)
        # Unsharp mask
        return np.clip(image + amount * (image - blurred), 0, 1)
    
    elif filter_type == "Frangi Filter (Vessel Enhancement)":
        from skimage.filters import frangi
        
        scale_range = st.sidebar.slider("Scale Range", 1, 10, (1, 3), 1)
        scale_step = st.sidebar.slider("Scale Step", 1, 10, 1, 1)
        
        return frangi(image, scale_range=scale_range, scale_step=scale_step)
    
    elif filter_type == "Gabor Filter":
        from skimage.filters import gabor
        
        frequency = st.sidebar.slider("Frequency", 0.01, 1.0, 0.1, 0.01)
        theta = st.sidebar.slider("Theta", 0.0, np.pi, np.pi/4, 0.1)
        sigma = st.sidebar.slider("Sigma", 0.5, 10.0, 3.0, 0.5)
        
        real, imag = gabor(image, frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
        return np.sqrt(real**2 + imag**2)
    
    elif filter_type == "Local Binary Pattern":
        from skimage.feature import local_binary_pattern
        
        p = st.sidebar.slider("Number of Points", 4, 24, 8, 4)
        r = st.sidebar.slider("Radius", 1, 10, 3, 1)
        method = st.sidebar.selectbox("Method", ["default", "uniform", "ror", "var"])
        
        lbp = local_binary_pattern(image, P=p, R=r, method=method)
        return (lbp - lbp.min()) / (lbp.max() - lbp.min())
    
    elif filter_type == "Gaussian Derivative":
        sigma = st.sidebar.slider("Sigma", 0.1, 10.0, 1.0, 0.1)
        order = st.sidebar.selectbox("Order", [0, 1, 2, 3])
        axis = st.sidebar.selectbox("Axis", ["x", "y", "both"])
        
        if axis == "x":
            return filters.gaussian(image, sigma=sigma, order=order, mode='nearest')
        elif axis == "y":
            return filters.gaussian(image, sigma=sigma, order=order, mode='nearest', axis=0)
        else:  # Both axes
            gx = filters.gaussian(image, sigma=sigma, order=order, mode='nearest')
            gy = filters.gaussian(image, sigma=sigma, order=order, mode='nearest', axis=0)
            return np.sqrt(gx**2 + gy**2)
    
    elif filter_type == "Homomorphic Filtering":
        gamma_low = st.sidebar.slider("Gamma Low", 0.1, 1.0, 0.5, 0.05)
        gamma_high = st.sidebar.slider("Gamma High", 1.0, 3.0, 1.5, 0.05)
        cutoff = st.sidebar.slider("Cutoff Frequency", 0.01, 0.5, 0.1, 0.01)
        
        # Take log of image to separate illumination and reflectance
        log_img = np.log1p(image)
        
        # Create frequency domain filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create meshgrid for filter
        u = np.linspace(-0.5, 0.5, cols)
        v = np.linspace(-0.5, 0.5, rows)
        U, V = np.meshgrid(u, v)
        D = np.sqrt(U**2 + V**2)
        
        # Create homomorphic filter
        H = (gamma_high - gamma_low) * (1 - np.exp(-D**2 / (2 * cutoff**2))) + gamma_low
        
        # Apply filter in frequency domain
        img_fft = np.fft.fftshift(np.fft.fft2(log_img))
        img_filtered = img_fft * H
        img_filtered = np.fft.ifft2(np.fft.ifftshift(img_filtered))
        
        # Take exponential to get back to spatial domain
        homomorphic = np.exp(np.real(img_filtered)) - 1
        
        # Normalize to [0,1]
        return (homomorphic - homomorphic.min()) / (homomorphic.max() - homomorphic.min())

def calculate_psnr(original, processed):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, processed):
    """Calculate Structural Similarity Index"""
    from skimage.metrics import structural_similarity as ssim
    return ssim(original, processed, data_range=processed.max() - processed.min())

# Import missing features for edge detection
from skimage import feature

if __name__ == "__main__":
    main()