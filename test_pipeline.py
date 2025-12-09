"""
Simple test script to verify OCR pipeline works without spaCy
Tests preprocessing and OCR modules on sample images
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath('./src'))

from preprocessing import preprocess_pipeline
from ocr_engine import extract_text, get_ocr_confidence

def test_pipeline(image_path):
    """Test the basic OCR pipeline"""
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*70}\n")
    
    try:
        # Step 1: Preprocess
        print("[1/3] Preprocessing image...")
        preprocessed = preprocess_pipeline(image_path)
        print("  ✓ Image preprocessed successfully")
        
        # Step 2: OCR
        print("[2/3] Extracting text with OCR...")
        text = extract_text(preprocessed['binary'])
        print(f"  ✓ Extracted {len(text)} characters")
        
        # Step 3: Get confidence
        print("[3/3] Calculating OCR confidence...")
        confidence = get_ocr_confidence(preprocessed['binary'])
        print(f"  ✓ Mean confidence: {confidence['mean_confidence']:.2f}%")
        print(f"  ✓ Word count: {confidence['word_count']}")
        
        print("\nExtracted Text (first 500 chars):")
        print("-" * 70)
        print(text[:500])
        print("-" * 70)
        
        return {
            'success': True,
            'text': text,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    # Test on all samples
    samples = [
        './Sample/page_14.jpg',
        './Sample/page_30.jpg',
        './Sample/page_35.jpg'
    ]
    
    print("\n" + "="*70)
    print(" " * 20 + "OCR PIPELINE TEST")
    print("="*70)
    
    results = {}
    for sample in samples:
        if os.path.exists(sample):
            results[sample] = test_pipeline(sample)
        else:
            print(f"Warning: {sample} not found")
    
    # Summary
    print("\n" + "="*70)
    print(" " * 25 + "SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results.values() if r.get('success'))
    print(f"\nTests passed: {successful}/{len(results)}")
    
    for sample, result in results.items():
        status = "✓ PASS" if result.get('success') else "✗ FAIL"
        print(f"  {status} - {os.path.basename(sample)}")
    
    print("\n" + "="*70)
