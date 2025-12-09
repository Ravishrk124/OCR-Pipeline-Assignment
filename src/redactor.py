"""
Redaction Module

This module generates redacted versions of original images by obscuring PII.
Maps text entities to image coordinates and applies visual redaction.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont


def map_text_to_coordinates(image, pii_entities):
    """
    Map PII text to image coordinates using OCR bounding boxes.
    
    Args:
        image (numpy.ndarray): Original or preprocessed image
        pii_entities (list): List of PII entities from pii_detector
        
    Returns:
        list: PII entities with added coordinate information
    """
    # Get OCR data with bounding boxes
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
    else:
        pil_image = image
    
    # Extract words with coordinates
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    
    # Build word-to-coordinate mapping
    word_positions = []
    current_text = ""
    
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word and int(data['conf'][i]) > 0:
            word_positions.append({
                'text': word,
                'x': data['left'][i],
                'y': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })
    
    # Match PII entities to coordinates
    entities_with_coords = []
    
    for entity in pii_entities:
        pii_text = entity['value'].strip()
        matched_boxes = []
        
        # Try to find matching words in OCR output
        for word_pos in word_positions:
            if word_pos['text'].lower() in pii_text.lower() or pii_text.lower() in word_pos['text'].lower():
                matched_boxes.append(word_pos)
        
        if matched_boxes:
            # Calculate bounding box encompassing all matched words
            x_min = min(box['x'] for box in matched_boxes)
            y_min = min(box['y'] for box in matched_boxes)
            x_max = max(box['x'] + box['width'] for box in matched_boxes)
            y_max = max(box['y'] + box['height'] for box in matched_boxes)
            
            entity_with_coords = entity.copy()
            entity_with_coords['coordinates'] = {
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            }
            entities_with_coords.append(entity_with_coords)
    
    return entities_with_coords


def create_redaction_boxes(image, entities_with_coords, color=(0, 0, 0), padding=2):
    """
    Draw black boxes over PII regions.
    
    Args:
        image (numpy.ndarray): Original image
        entities_with_coords (list): PII entities with coordinate information
        color (tuple): BGR color for redaction boxes (default: black)
        padding (int): Padding around text boxes
        
    Returns:
        numpy.ndarray: Redacted image
    """
    redacted = image.copy()
    
    for entity in entities_with_coords:
        if 'coordinates' in entity:
            coords = entity['coordinates']
            x = max(0, coords['x'] - padding)
            y = max(0, coords['y'] - padding)
            x2 = coords['x'] + coords['width'] + padding
            y2 = coords['y'] + coords['height'] + padding
            
            # Draw filled rectangle
            cv2.rectangle(redacted, (x, y), (x2, y2), color, -1)
    
    return redacted


def blur_pii_regions(image, entities_with_coords, blur_strength=25, padding=2):
    """
    Apply Gaussian blur over PII regions (alternative to black boxes).
    
    Args:
        image (numpy.ndarray): Original image
        entities_with_coords (list): PII entities with coordinate information
        blur_strength (int): Strength of Gaussian blur (must be odd)
        padding (int): Padding around text boxes
        
    Returns:
        numpy.ndarray: Blurred image
    """
    redacted = image.copy()
    
    # Ensure blur strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    for entity in entities_with_coords:
        if 'coordinates' in entity:
            coords = entity['coordinates']
            x = max(0, coords['x'] - padding)
            y = max(0, coords['y'] - padding)
            x2 = min(image.shape[1], coords['x'] + coords['width'] + padding)
            y2 = min(image.shape[0], coords['y'] + coords['height'] + padding)
            
            # Extract region
            roi = redacted[y:y2, x:x2]
            
            # Apply Gaussian blur
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
                redacted[y:y2, x:x2] = blurred_roi
    
    return redacted


def create_labeled_redaction(image, entities_with_coords, padding=2):
    """
    Create redaction with labels showing PII type.
    
    Args:
        image (numpy.ndarray): Original image
        entities_with_coords (list): PII entities with coordinate information
        padding (int): Padding around text boxes
        
    Returns:
        numpy.ndarray: Labeled redacted image
    """
    redacted = image.copy()
    
    # Color mapping for different PII types
    colors = {
        'PERSON': (0, 0, 255),      # Red
        'PHONE': (255, 0, 0),       # Blue
        'EMAIL': (0, 255, 255),     # Yellow
        'DATE': (0, 255, 0),        # Green
        'ADDRESS': (255, 0, 255),   # Magenta
        'MEDICAL_ID': (128, 0, 128), # Purple
        'ORG': (255, 165, 0)        # Orange
    }
    
    for entity in entities_with_coords:
        if 'coordinates' in entity:
            coords = entity['coordinates']
            x = max(0, coords['x'] - padding)
            y = max(0, coords['y'] - padding)
            x2 = coords['x'] + coords['width'] + padding
            y2 = coords['y'] + coords['height'] + padding
            
            # Get color for PII type
            color = colors.get(entity['type'], (0, 0, 0))
            
            # Draw semi-transparent rectangle
            overlay = redacted.copy()
            cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.6, redacted, 0.4, 0, redacted)
            
            # Draw border
            cv2.rectangle(redacted, (x, y), (x2, y2), color, 2)
            
            # Add label
            label = entity['type']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            
            # Get label size
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw label background
            cv2.rectangle(redacted, (x, y - label_h - 4), (x + label_w + 4, y), color, -1)
            
            # Draw label text
            cv2.putText(redacted, label, (x + 2, y - 2), font, font_scale, (255, 255, 255), font_thickness)
    
    return redacted


def generate_redacted_image(image_path, pii_data, output_path=None, redaction_type='black'):
    """
    Complete redaction workflow.
    
    Args:
        image_path (str): Path to original image
        pii_data (dict): PII detection results
        output_path (str, optional): Path to save redacted image
        redaction_type (str): Type of redaction ('black', 'blur', 'labeled')
        
    Returns:
        dict: Dictionary with original and redacted images
    """
    # Load original image
    original = cv2.imread(image_path)
    
    # Map PII to coordinates
    entities_with_coords = map_text_to_coordinates(original, pii_data['pii_entities'])
    
    # Apply redaction based on type
    if redaction_type == 'blur':
        redacted = blur_pii_regions(original, entities_with_coords)
    elif redaction_type == 'labeled':
        redacted = create_labeled_redaction(original, entities_with_coords)
    else:  # 'black'
        redacted = create_redaction_boxes(original, entities_with_coords)
    
    # Save if path provided
    if output_path:
        cv2.imwrite(output_path, redacted)
    
    return {
        'original': original,
        'redacted': redacted,
        'entities_redacted': len(entities_with_coords),
        'redaction_type': redaction_type
    }


def create_comparison_image(original, redacted):
    """
    Create side-by-side comparison of original and redacted images.
    
    Args:
        original (numpy.ndarray): Original image
        redacted (numpy.ndarray): Redacted image
        
    Returns:
        numpy.ndarray: Side-by-side comparison
    """
    # Ensure both images have same height
    h1, w1 = original.shape[:2]
    h2, w2 = redacted.shape[:2]
    
    max_h = max(h1, h2)
    
    # Resize if needed to match heights
    if h1 != max_h:
        scale = max_h / h1
        original = cv2.resize(original, (int(w1 * scale), max_h))
    if h2 != max_h:
        scale = max_h / h2
        redacted = cv2.resize(redacted, (int(w2 * scale), max_h))
    
    # Create comparison image
    comparison = np.hstack([original, redacted])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (20, 40), font, 1.5, (0, 255, 0), 3)
    cv2.putText(comparison, "Redacted", (original.shape[1] + 20, 40), font, 1.5, (0, 0, 255), 3)
    
    return comparison
