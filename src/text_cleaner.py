"""
Text Cleaning Module

This module provides functions to clean and normalize OCR output.
Handles common OCR errors and noise characters.
"""

import re
import string


def remove_noise_characters(text):
    """
    Remove OCR artifacts and gibberish characters.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    # Remove multiple special characters in sequence
    text = re.sub(r'[^\w\s\.,;:!?\-()/@#$%&*+=\[\]{}\'\"]+', '', text)
    
    # Remove standalone special characters (not part of words)
    text = re.sub(r'\s[^\w\s]\s', ' ', text)
    
    # Remove excessive special characters
    text = re.sub(r'([^\w\s])\1{2,}', r'\1', text)
    
    return text


def normalize_whitespace(text):
    """
    Standardize spacing and line breaks.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove spaces at start/end of lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def correct_common_ocr_errors(text):
    """
    Fix typical OCR mistakes.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Corrected text
    """
    # Common OCR confusions
    corrections = {
        r'\b0\b': 'O',  # Standalone 0 -> O
        r'\bl\b': 'I',  # Standalone l -> I
        r'(?<=[a-z])0(?=[a-z])': 'o',  # 0 in middle of word -> o
        r'(?<=[A-Z])0(?=[A-Z])': 'O',  # 0 in middle of uppercase -> O
        r'(?<=\d)O(?=\d)': '0',  # O between digits -> 0
        r'(?<=\d)l(?=\d)': '1',  # l between digits -> 1
        r'(?<=\d)I(?=\d)': '1',  # I between digits -> 1
        r'rn': 'm',  # rn -> m (common OCR error)
        r'vv': 'w',  # vv -> w
        r'\|\|': 'll',  # || -> ll
    }
    
    # Apply corrections
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    
    return text


def standardize_dates(text):
    """
    Standardize date formats.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with standardized dates
    """
    # Find various date formats and standardize
    # DD/MM/YYYY or DD-MM-YYYY -> DD/MM/YYYY
    text = re.sub(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', r'\1/\2/\3', text)
    
    # MM/DD/YY -> MM/DD/20YY
    text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{2})\b', r'\1/\2/20\3', text)
    
    return text


def standardize_phone_numbers(text):
    """
    Standardize phone number formats.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with standardized phone numbers
    """
    # Remove common OCR errors in phone numbers
    # Fix O -> 0 in phone numbers
    text = re.sub(r'(\(\d{3}\)|^\d{3})\s*[O]', r'\g<1> 0', text)
    
    return text


def remove_extra_punctuation(text):
    """
    Remove excessive or misplaced punctuation.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    # Remove multiple punctuation (except ...)
    text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before
    text = re.sub(r'([.,!?;:])(?=[a-zA-Z])', r'\1 ', text)  # Add space after
    
    return text


def clean_pipeline(text):
    """
    Complete text cleaning workflow.
    
    Args:
        text (str): Raw OCR output
        
    Returns:
        str: Cleaned and normalized text
    """
    # Step 1: Remove noise characters
    text = remove_noise_characters(text)
    
    # Step 2: Normalize whitespace
    text = normalize_whitespace(text)
    
    # Step 3: Correct common OCR errors
    text = correct_common_ocr_errors(text)
    
    # Step 4: Standardize dates
    text = standardize_dates(text)
    
    # Step 5: Standardize phone numbers
    text = standardize_phone_numbers(text)
    
    # Step 6: Remove extra punctuation
    text = remove_extra_punctuation(text)
    
    # Step 7: Final whitespace normalization
    text = normalize_whitespace(text)
    
    return text


def get_cleaning_stats(original_text, cleaned_text):
    """
    Get statistics about cleaning process.
    
    Args:
        original_text (str): Original text
        cleaned_text (str): Cleaned text
        
    Returns:
        dict: Cleaning statistics
    """
    return {
        'original_length': len(original_text),
        'cleaned_length': len(cleaned_text),
        'characters_removed': len(original_text) - len(cleaned_text),
        'original_words': len(original_text.split()),
        'cleaned_words': len(cleaned_text.split()),
        'original_lines': len(original_text.split('\n')),
        'cleaned_lines': len(cleaned_text.split('\n'))
    }
