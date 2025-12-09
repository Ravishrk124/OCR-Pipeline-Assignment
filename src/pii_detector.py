"""
PII Detection Module

This module identifies Personally Identifiable Information (PII) in text.
Detects: names, phone numbers, emails, dates, addresses, medical IDs.
"""

import re
import spacy
from typing import List, Dict, Any


# Load spaCy model (will be loaded on first use)
_nlp = None


def setup_spacy_model():
    """
    Load spaCy NER model.
    
    Returns:
        spacy.Language: Loaded spaCy model
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, provide helpful error
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Please install it using: python -m spacy download en_core_web_sm"
            )
    return _nlp


def detect_names(text):
    """
    Detect person names using spaCy NER.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of detected names with positions
    """
    nlp = setup_spacy_model()
    doc = nlp(text)
    
    names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append({
                'type': 'PERSON',
                'value': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.85  # spaCy NER is fairly reliable
            })
    
    return names


def detect_phone_numbers(text):
    """
    Detect phone numbers using regex patterns.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of detected phone numbers with positions
    """
    phone_patterns = [
        # (123) 456-7890
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
        # 123-456-7890
        r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',
        # 1234567890
        r'\b\d{10}\b',
        # +1 123 456 7890
        r'\+\d{1,3}\s?\d{3}\s?\d{3}\s?\d{4}',
        # India format: +91 12345 67890
        r'\+91\s?\d{5}\s?\d{5}',
        # India format: 12345-67890
        r'\d{5}[-.\s]?\d{5}',
    ]
    
    phones = []
    for pattern in phone_patterns:
        for match in re.finditer(pattern, text):
            phones.append({
                'type': 'PHONE',
                'value': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
    
    return phones


def detect_emails(text):
    """
    Detect email addresses using regex.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of detected emails with positions
    """
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    emails = []
    for match in re.finditer(email_pattern, text):
        emails.append({
            'type': 'EMAIL',
            'value': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': 0.95
        })
    
    return emails


def detect_dates(text):
    """
    Detect dates using spaCy and regex.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of detected dates with positions
    """
    dates = []
    
    # Use spaCy for DATE entities
    nlp = setup_spacy_model()
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.append({
                'type': 'DATE',
                'value': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.8
            })
    
    # Also use regex for common date formats
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY, DD/MM/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY, DD-MM-YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Check if already detected by spaCy
            if not any(d['start'] == match.start() for d in dates):
                dates.append({
                    'type': 'DATE',
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85
                })
    
    return dates


def detect_addresses(text):
    """
    Detect physical addresses and locations using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of detected addresses with positions
    """
    nlp = setup_spacy_model()
    doc = nlp(text)
    
    addresses = []
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:  # Geopolitical entities, locations, facilities
            addresses.append({
                'type': 'ADDRESS',
                'value': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.75
            })
    
    # Also detect ZIP codes
    zip_pattern = r'\b\d{5}(?:-\d{4})?\b'
    for match in re.finditer(zip_pattern, text):
        addresses.append({
            'type': 'ADDRESS',
            'value': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': 0.9
        })
    
    return addresses


def detect_medical_ids(text):
    """
    Detect medical record numbers and patient IDs.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of detected medical IDs with positions
    """
    medical_ids = []
    
    # Patterns for medical IDs
    patterns = [
        # Patient ID, Medical Record Number
        r'\b(?:Patient\s+ID|MRN|Medical\s+Record\s+Number|Record\s+#)\s*:?\s*([A-Z0-9-]+)\b',
        # Generic alphanumeric IDs (6+ characters)
        r'\b[A-Z]{2}\d{6,}\b',
        r'\b\d{6,10}\b',  # Numeric IDs
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            medical_ids.append({
                'type': 'MEDICAL_ID',
                'value': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.7
            })
    
    return medical_ids


def detect_organizations(text):
    """
    Detect organization names (hospitals, clinics).
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of detected organizations with positions
    """
    nlp = setup_spacy_model()
    doc = nlp(text)
    
    organizations = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            organizations.append({
                'type': 'ORG',
                'value': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.8
            })
    
    return organizations


def merge_overlapping_entities(entities):
    """
    Merge overlapping PII entities (keep higher confidence).
    
    Args:
        entities (list): List of PII entities
        
    Returns:
        list: Deduplicated entities
    """
    if not entities:
        return []
    
    # Sort by start position
    sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['confidence']))
    
    merged = []
    current = sorted_entities[0]
    
    for entity in sorted_entities[1:]:
        # Check for overlap
        if entity['start'] < current['end']:
            # Keep entity with higher confidence
            if entity['confidence'] > current['confidence']:
                current = entity
        else:
            merged.append(current)
            current = entity
    
    merged.append(current)
    
    return merged


def detect_all_pii(text):
    """
    Comprehensive PII detection with all categories.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with original text and detected PII entities
    """
    all_entities = []
    
    # Detect all PII types
    all_entities.extend(detect_names(text))
    all_entities.extend(detect_phone_numbers(text))
    all_entities.extend(detect_emails(text))
    all_entities.extend(detect_dates(text))
    all_entities.extend(detect_addresses(text))
    all_entities.extend(detect_medical_ids(text))
    all_entities.extend(detect_organizations(text))
    
    # Merge overlapping entities
    unique_entities = merge_overlapping_entities(all_entities)
    
    # Sort by position
    unique_entities.sort(key=lambda x: x['start'])
    
    return {
        'text': text,
        'pii_entities': unique_entities,
        'pii_count': len(unique_entities),
        'pii_types': {
            'PERSON': len([e for e in unique_entities if e['type'] == 'PERSON']),
            'PHONE': len([e for e in unique_entities if e['type'] == 'PHONE']),
            'EMAIL': len([e for e in unique_entities if e['type'] == 'EMAIL']),
            'DATE': len([e for e in unique_entities if e['type'] == 'DATE']),
            'ADDRESS': len([e for e in unique_entities if e['type'] == 'ADDRESS']),
            'MEDICAL_ID': len([e for e in unique_entities if e['type'] == 'MEDICAL_ID']),
            'ORG': len([e for e in unique_entities if e['type'] == 'ORG']),
        }
    }


def highlight_pii_in_text(text, pii_data, highlight_char='█'):
    """
    Create a text version with PII highlighted/redacted.
    
    Args:
        text (str): Original text
        pii_data (dict): PII detection results
        highlight_char (str): Character to use for highlighting
        
    Returns:
        str: Text with PII highlighted
    """
    result = text
    offset = 0
    
    for entity in pii_data['pii_entities']:
        start = entity['start'] + offset
        end = entity['end'] + offset
        
        # Create redaction
        redaction = highlight_char * (end - start)
        
        # Replace in result
        result = result[:start] + redaction + result[end:]
        
        # No offset change since replacement is same length
    
    return result


def get_pii_summary(pii_data):
    """
    Get a summary of detected PII.
    
    Args:
        pii_data (dict): PII detection results
        
    Returns:
        str: Formatted summary
    """
    summary = f"Total PII Entities Detected: {pii_data['pii_count']}\n\n"
    summary += "Breakdown by Type:\n"
    
    for pii_type, count in pii_data['pii_types'].items():
        if count > 0:
            summary += f"  - {pii_type}: {count}\n"
    
    summary += "\nDetected Entities:\n"
    for entity in pii_data['pii_entities']:
        summary += f"  - {entity['type']}: '{entity['value']}' (confidence: {entity['confidence']:.2f})\n"
    
    return summary
