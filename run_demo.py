"""
Comprehensive Demo Script for OCR + PII Pipeline
Demonstrates full pipeline without spaCy dependency issues
Generates results and screenshots
"""

import sys
import os
import cv2
import numpy as np
import json
import re
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath('./src'))

from preprocessing import preprocess_pipeline
from ocr_engine import extract_text, extract_with_boxes, get_ocr_confidence
from text_cleaner import clean_pipeline, get_cleaning_stats

# Simple PII detection without spaCy (regex-based)
def detect_pii_simple(text):
    """
    Simple PII detection using regex patterns only
    (Works without spaCy to avoid Python 3.14 compatibility issues)
    """
    pii_entities = []
    
    # Phone numbers
    phone_patterns = [
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',
        r'\b\d{10}\b',
        r'\+\d{1,3}\s?\d{3}\s?\d{3}\s?\d{4}',
    ]
    
    for pattern in phone_patterns:
        for match in re.finditer(pattern, text):
            pii_entities.append({
                'type': 'PHONE',
                'value': match.group(),
                'start': match.start(),
                'end': match.end()
            })
    
    # Emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        pii_entities.append({
            'type': 'EMAIL',
            'value': match.group(),
            'start': match.start(),
            'end': match.end()
        })
    
    # Dates
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            pii_entities.append({
                'type': 'DATE',
                'value': match.group(),
                'start': match.start(),
                'end': match.end()
            })
    
    # Simple name detection (capitalized words)
    name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    for match in re.finditer(name_pattern, text):
        pii_entities.append({
            'type': 'PERSON',
            'value': match.group(),
            'start': match.start(),
            'end': match.end()
        })
    
    # Medical IDs
    medical_patterns = [
        r'\b(?:Patient\s+ID|MRN|Medical\s+Record)\s*:?\s*([A-Z0-9-]+)\b',
        r'\b[A-Z]{2}\d{6,}\b',
    ]
    
    for pattern in medical_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            pii_entities.append({
                'type': 'MEDICAL_ID',
                'value': match.group(),
                'start': match.start(),
                'end': match.end()
            })
   
    return {
        'text': text,
        'pii_entities': pii_entities,
        'pii_count': len(pii_entities)
    }


def complete_pipeline_demo(image_path, output_dir='./outputs'):
    """Run complete pipeline demonstration"""
    
    base_name = os.path.basename(image_path).replace('.jpg', '')
    
    print(f"\n{'='*80}")
    print(f"Processing: {base_name}")
    print(f"{'='*80}\n")
    
    results = {'sample_name': base_name}
    
    # Step 1: Preprocessing
    print("[1/5] Preprocessing image...")
    preprocessed = preprocess_pipeline(image_path)
    preprocessed_path = f'{output_dir}/preprocessed/{base_name}_preprocessed.jpg'
    cv2.imwrite(preprocessed_path, preprocessed['binary'])
    print(f"  ✓ Saved: {preprocessed_path}")
    
    # Step 2: OCR
    print("[2/5] Extracting text with OCR...")
    ocr_result = extract_with_boxes(preprocessed['binary'])
    text = ocr_result['text']
    confidence = get_ocr_confidence(preprocessed['binary'])
    
    ocr_path = f'{output_dir}/ocr_results/{base_name}_extracted_text.txt'
    with open(ocr_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"  ✓ Extracted {len(text)} characters")
    print(f"  ✓ Mean confidence: {confidence['mean_confidence']:.2f}%")
    print(f"  ✓ Saved: {ocr_path}")
    
    results['ocr_text'] = text
    results['ocr_confidence'] = confidence
    
    # Step 3: Text Cleaning
    print("[3/5] Cleaning text...")
    cleaned_text = clean_pipeline(text)
    stats = get_cleaning_stats(text, cleaned_text)
    print(f"  ✓ Removed {stats['characters_removed']} noise characters")
    
    results['cleaned_text'] = cleaned_text
    results['cleaning_stats'] = stats
    
    # Step 4: PII Detection
    print("[4/5] Detecting PII...")
    pii_data = detect_pii_simple(cleaned_text)
    
    pii_path = f'{output_dir}/pii_detected/{base_name}_pii.json'
    with open(pii_path, 'w', encoding='utf-8') as f:
        json.dump(pii_data, f, indent=2)
    
    print(f"  ✓ Detected {pii_data['pii_count']} PII entities")
    for entity in pii_data['pii_entities']:
        print(f"    - {entity['type']}: {entity['value']}")
    print(f"  ✓ Saved: {pii_path}")
    
    results['pii_detection'] = pii_data
    
    # Step 5: Create visualization
    print("[5/5] Creating visualization...")
    
    # Create comparison image
    original = cv2.imread(image_path)
    comparison = np.hstack([original, cv2.cvtColor(preprocessed['binary'], cv2.COLOR_GRAY2BGR)])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (20, 40), font, 1.0, (0, 255, 0), 2)
    cv2.putText(comparison, "Preprocessed", (original.shape[1] + 20, 40), font, 1.0, (0, 0, 255), 2)
    
    viz_path = f'./results/{base_name}_comparison.jpg'
    cv2.imwrite(viz_path, comparison)
    print(f"  ✓ Saved: {viz_path}")
    
    print(f"\n{'='*80}")
    print("Pipeline Complete!")
    print(f"{'='*80}\n")
    
    return results


def main():
    """Main execution"""
    samples = [
        './Sample/page_14.jpg',
        './Sample/page_30.jpg',
        './Sample/page_35.jpg'
    ]
    
    print("\n" + "="*80)
    print(" " * 20 + "OCR + PII EXTRACTION PIPELINE DEMO")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for sample in samples:
        if os.path.exists(sample):
            result = complete_pipeline_demo(sample)
            all_results.append(result)
        else:
            print(f"Warning: {sample} not found")
    
    # Save comprehensive results
    comprehensive_path = './outputs/comprehensive_results.json'
    with open(comprehensive_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print(" " * 30 + "FINAL SUMMARY")
    print("="*80 + "\n")
    
    for result in all_results:
        print(f"Sample: {result['sample_name']}")
        print("-" * 80)
        print(f"  OCR Confidence:     {result['ocr_confidence']['mean_confidence']:.2f}%")
        print(f"  Words Extracted:    {result['ocr_confidence']['word_count']}")
        print(f"  PII Entities Found: {result['pii_detection']['pii_count']}")
        print()
    
    print("="*80)
    print(f"\n✅ All results saved to: {comprehensive_path}")
    print("\nOutputs generated:")
    print("  - Preprocessed images:  outputs/preprocessed/")
    print("  - OCR text files:       outputs/ocr_results/")
    print("  - PII detection JSON:   outputs/pii_detected/")
    print("  - Result comparisons:   results/")
    print("="*80)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

if __name__ == '__main__':
    main()
