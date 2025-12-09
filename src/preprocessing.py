"""
Image Preprocessing Module for OCR Pipeline

This module provides functions to enhance image quality for better OCR accuracy.
Handles: rotation correction, noise reduction, contrast enhancement, and binarization.
"""

import cv2
import numpy as np
from PIL import Image


def load_image(image_path):
    """
    Load and validate JPEG image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image in BGR format
        
    Raises:
        ValueError: If image cannot be loaded
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img


def detect_rotation(image):
    """
    Detect image skew/tilt angle using Hough transform.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        float: Rotation angle in degrees
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is None:
        return 0.0
    
    # Calculate angles
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        angles.append(angle)
    
    # Return median angle to avoid outliers
    if angles:
        median_angle = np.median(angles)
        # Only significant tilts
        if abs(median_angle) > 0.5:
            return median_angle
    
    return 0.0


def deskew_image(image, angle=None):
    """
    Correct tilted/rotated images.
    
    Args:
        image (numpy.ndarray): Input image
        angle (float, optional): Rotation angle. If None, auto-detect.
        
    Returns:
        numpy.ndarray: Deskewed image
    """
    if angle is None:
        angle = detect_rotation(image)
    
    if abs(angle) < 0.5:
        return image
    
    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def denoise_image(image):
    """
    Apply bilateral filtering for noise reduction while preserving edges.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Denoised image
    """
    # Bilateral filter: reduces noise while keeping edges sharp
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised


def enhance_contrast(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Contrast-enhanced image
    """
    # Convert to LAB color space
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
    else:
        l = image.copy()
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge back
    if len(image.shape) == 3:
        enhanced_lab = cv2.merge([cl, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        enhanced = cl
    
    return enhanced


def binarize_image(image):
    """
    Convert image to binary (black and white) using adaptive thresholding.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Binarized image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Adaptive Gaussian thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    return binary


def preprocess_pipeline(image_path, save_path=None):
    """
    Complete pre-processing workflow for handwritten documents.
    
    Args:
        image_path (str): Path to input image
        save_path (str, optional): Path to save preprocessed image
        
    Returns:
        dict: Dictionary containing all processed versions
            - 'original': Original image
            - 'deskewed': Rotation-corrected image
            - 'denoised': Noise-reduced image
            - 'enhanced': Contrast-enhanced image
            - 'binary': Final binarized image
    """
    # Load image
    original = load_image(image_path)
    
    # Step 1: Deskew
    deskewed = deskew_image(original)
    
    # Step 2: Denoise
    denoised = denoise_image(deskewed)
    
    # Step 3: Enhance contrast
    enhanced = enhance_contrast(denoised)
    
    # Step 4: Binarize
    binary = binarize_image(enhanced)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, binary)
    
    return {
        'original': original,
        'deskewed': deskewed,
        'denoised': denoised,
        'enhanced': enhanced,
        'binary': binary
    }


def visualize_preprocessing_steps(image_path):
    """
    Visualize all preprocessing steps side-by-side.
    
    Args:
        image_path (str): Path to input image
        
    Returns:
        dict: All preprocessing stages
    """
    import matplotlib.pyplot as plt
    
    results = preprocess_pipeline(image_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Image Preprocessing Pipeline', fontsize=16)
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Original')
    axes[0, 0].axis('off')
    
    # Deskewed
    axes[0, 1].imshow(cv2.cvtColor(results['deskewed'], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('2. Deskewed')
    axes[0, 1].axis('off')
    
    # Denoised
    axes[0, 2].imshow(cv2.cvtColor(results['denoised'], cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('3. Denoised')
    axes[0, 2].axis('off')
    
    # Enhanced
    axes[1, 0].imshow(cv2.cvtColor(results['enhanced'], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('4. Contrast Enhanced')
    axes[1, 0].axis('off')
    
    # Binary
    axes[1, 1].imshow(results['binary'], cmap='gray')
    axes[1, 1].set_title('5. Binarized (Final)')
    axes[1, 1].axis('off')
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    return results, fig
