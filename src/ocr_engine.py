"""
OCR Engine Module

This module handles text extraction from preprocessed images using Tesseract OCR.
Optimized for handwritten text recognition.
"""

import pytesseract
import cv2
import numpy as np
from PIL import Image
import re


def configure_tesseract():
    """
    Get optimal Tesseract configuration for handwriting recognition.
    
    Returns:
        str: Tesseract configuration string
    """
    # PSM (Page Segmentation Mode):
    # 3 = Fully automatic page segmentation (default)
    # 6 = Uniform block of text
    # OEM (OCR Engine Mode):
    # 3 = Default, based on what is available (LSTM + Legacy)
    config = '--oem 3 --psm 3'
    return config


def extract_text(image, config=None):
    """
    Extract text from preprocessed image.
    
    Args:
        image (numpy.ndarray): Preprocessed image
        config (str, optional): Tesseract configuration
        
    Returns:
        str: Extracted text
    """
    if config is None:
        config = configure_tesseract()
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        # Handle different image formats
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image)
        else:  # BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
    else:
        pil_image = image
    
    # Extract text
    text = pytesseract.image_to_string(pil_image, config=config)
    
    return text


def extract_with_boxes(image, config=None):
    """
    Extract text with bounding box coordinates (for redaction).
    
    Args:
        image (numpy.ndarray): Preprocessed image
        config (str, optional): Tesseract configuration
        
    Returns:
        dict: Dictionary with 'text' and 'boxes' (list of word boxes)
    """
    if config is None:
        config = configure_tesseract()
    
    # Convert to PIL Image
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
    else:
        pil_image = image
    
    # Get OCR data with bounding boxes
    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
    
    # Extract words and their bounding boxes
    boxes = []
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        # Filter out empty detections
        if int(data['conf'][i]) > 0:  # Confidence > 0
            word = data['text'][i].strip()
            if word:  # Non-empty word
                boxes.append({
                    'text': word,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'confidence': data['conf'][i]
                })
    
    # Extract full text
    text = pytesseract.image_to_string(pil_image, config=config)
    
    return {
        'text': text,
        'boxes': boxes
    }


def multi_pass_ocr(image):
    """
    Try multiple PSM modes and return best result.
    
    Args:
        image (numpy.ndarray): Preprocessed image
        
    Returns:
        dict: Best OCR result with text and confidence
    """
    psm_modes = [3, 6, 4, 11]  # Different page segmentation modes
    results = []
    
    for psm in psm_modes:
        config = f'--oem 3 --psm {psm}'
        try:
            text = extract_text(image, config=config)
            # Simple heuristic: longer text with more words is likely better
            word_count = len(text.split())
            results.append({
                'text': text,
                'psm': psm,
                'word_count': word_count,
                'length': len(text.strip())
            })
        except Exception as e:
            continue
    
    # Return result with most words
    if results:
        best_result = max(results, key=lambda x: x['word_count'])
        return best_result
    else:
        return {'text': '', 'psm': 3, 'word_count': 0, 'length': 0}


def get_ocr_confidence(image, config=None):
    """
    Get OCR confidence scores.
    
    Args:
        image (numpy.ndarray): Preprocessed image
        config (str, optional): Tesseract configuration
        
    Returns:
        dict: Confidence statistics
    """
    if config is None:
        config = configure_tesseract()
    
    # Convert to PIL Image
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
    else:
        pil_image = image
    
    # Get detailed OCR data
    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
    
    # Calculate confidence statistics
    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
    
    if confidences:
        return {
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'word_count': len(confidences)
        }
    else:
        return {
            'mean_confidence': 0,
            'median_confidence': 0,
            'min_confidence': 0,
            'max_confidence': 0,
            'word_count': 0
        }


def ocr_pipeline(image, extract_boxes=False, get_confidence=False):
    """
    Complete OCR pipeline with all options.
    
    Args:
        image (numpy.ndarray): Preprocessed image
        extract_boxes (bool): Whether to extract bounding boxes
        get_confidence (bool): Whether to calculate confidence scores
        
    Returns:
        dict: OCR results with requested information
    """
    result = {}
    
    # Extract text
    if extract_boxes:
        data = extract_with_boxes(image)
        result['text'] = data['text']
        result['boxes'] = data['boxes']
    else:
        result['text'] = extract_text(image)
    
    # Get confidence if requested
    if get_confidence:
        result['confidence'] = get_ocr_confidence(image)
    
    return result
