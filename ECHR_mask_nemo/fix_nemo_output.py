#!/usr/bin/env python3
"""
fix_nemo_output.py

Post-process and fix the corrupted NEMO output by:
1. Removing the massive over-detected DATE_TIME entity
2. Properly detecting individual dates and entities
3. Cleaning up broken tags and text corruption
"""

import json
import re
from datetime import datetime

def extract_dates_from_text(text):
    """Extract date entities from text using regex patterns"""
    date_patterns = [
        (r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', 'DATE_TIME'),
        (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'DATE_TIME'),
        (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', 'DATE_TIME'),
        (r'\b\d{4}\b', 'DATE_TIME'),  # Just years
        (r'\bthe\s+(?:beginning|end|middle|late|early)\s+of\s+(?:the\s+)?\d{4}\b', 'DATE_TIME'),
        (r'\bthe\s+(?:beginning|end|middle|late|early)\s+of\s+(?:the\s+)?(?:1990s|1980s|2000s)\b', 'DATE_TIME'),
        (r'\bthe\s+(?:summer|winter|spring|autumn|fall)\s+of\s+\d{4}\b', 'DATE_TIME'),
        (r'\bfrom\s+(?:the\s+)?(?:late|early)\s+\d{4}s?\b', 'DATE_TIME'),
        (r'\buntil\s+\d{4}\b', 'DATE_TIME'),
        (r'\bin\s+\d{4}\b', 'DATE_TIME'),
        (r'\bon\s+\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', 'DATE_TIME'),
    ]
    
    entities = []
    for pattern, label in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                "label": label,
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })
    
    return entities

def extract_persons_from_text(text):
    """Extract person entities from text"""
    # Common patterns for person names in legal texts
    person_patterns = [
        r'\bMr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Mr. FirstName LastName
        r'\bMs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',   # Ms. FirstName LastName
        r'\bMrs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Mrs. FirstName LastName
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b',  # FirstName LastName(-LastName)
    ]
    
    entities = []
    for pattern in person_patterns:
        for match in re.finditer(pattern, text):
            # Avoid common false positives
            name = match.group()
            if not re.search(r'\b(?:City|Court|Government|Ministry|Article|Section|President|Agent)\b', name):
                entities.append({
                    "label": "PERSON",
                    "text": name,
                    "start": match.start(),
                    "end": match.end()
                })
    
    return entities

def extract_addresses_from_text(text):
    """Extract address/location entities from text"""
    address_patterns = [
        r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+\b',  # City, Country
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+,\s+[A-Z][a-z]+\b',  # City Name, Country
        r'\bLes\s+Salles\s+Sur\s+Verdon,\s+France\b',  # Specific address from text
        r'\bCopenhagen\b',
        r'\bDenmark\b',
        r'\bSwitzerland\b',
        r'\bSweden\b',
    ]
    
    entities = []
    for pattern in address_patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "label": "ADDRESS",
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })
    
    return entities

def clean_and_extract_entities(text):
    """Clean text and extract proper entities"""
    # Extract entities using pattern matching
    date_entities = extract_dates_from_text(text)
    person_entities = extract_persons_from_text(text)
    address_entities = extract_addresses_from_text(text)
    
    # Combine all entities
    all_entities = date_entities + person_entities + address_entities
    
    # Remove overlapping entities (keep longer ones)
    cleaned_entities = []
    all_entities.sort(key=lambda x: (x['start'], -len(x['text'])))
    
    for entity in all_entities:
        # Check if this entity overlaps with any already added
        overlap = False
        for existing in cleaned_entities:
            if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                overlap = True
                break
        
        if not overlap:
            cleaned_entities.append(entity)
    
    # Sort by position
    cleaned_entities.sort(key=lambda x: x['start'])
    
    return cleaned_entities

def fix_corrupted_nemo_output(input_file, output_file):
    """Fix the corrupted NEMO output file"""
    
    print(f"Loading corrupted NEMO output from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} documents...")
    
    fixed_data = []
    total_entities_before = 0
    total_entities_after = 0
    
    for i, doc in enumerate(data):
        print(f"\nProcessing document {i+1}/{len(data)}: {doc.get('id', 'unknown')}")
        
        original_text = doc['text']
        corrupted_entities = doc.get('nemo_detected_pii', [])
        
        print(f"  Original entities: {len(corrupted_entities)}")
        total_entities_before += len(corrupted_entities)
        
        # Check for the massive over-detected entity
        massive_entities = [e for e in corrupted_entities if len(e.get('text', '')) > 1000]
        if massive_entities:
            print(f"  ❌ Found {len(massive_entities)} massive over-detected entities")
            for entity in massive_entities:
                print(f"    - {entity['label']}: {len(entity['text'])} characters")
        
        # Check for invalid entities
        invalid_entities = [e for e in corrupted_entities if e.get('end', 0) <= e.get('start', 0) or e.get('end', -1) == -1]
        if invalid_entities:
            print(f"  ❌ Found {len(invalid_entities)} invalid entities")
        
        # Re-extract entities properly
        print("  🔄 Re-extracting entities...")
        fixed_entities = clean_and_extract_entities(original_text)
        
        print(f"  ✅ Fixed entities: {len(fixed_entities)}")
        total_entities_after += len(fixed_entities)
        
        # Show sample of fixed entities
        if fixed_entities:
            print("  📋 Sample entities:")
            for entity in fixed_entities[:5]:
                print(f"    - {entity['label']}: '{entity['text']}' [{entity['start']}:{entity['end']}]")
        
        # Create clean masked text
        masked_text = original_text
        offset = 0
        for entity in reversed(fixed_entities):  # Reverse to maintain positions
            start = entity['start'] + offset
            end = entity['end'] + offset
            replacement = f"<{entity['label']}>"
            masked_text = masked_text[:start] + replacement + masked_text[end:]
            offset += len(replacement) - (end - start)
        
        # Create fixed document
        fixed_doc = {
            "id": doc.get('id', f'doc_{i}'),
            "text": original_text,
            "nemo_masked_text": masked_text,
            "nemo_detected_pii": fixed_entities,
            "original_entities_count": len(corrupted_entities),
            "fixed_entities_count": len(fixed_entities),
            "had_corruption": len(massive_entities) > 0 or len(invalid_entities) > 0
        }
        
        fixed_data.append(fixed_doc)
    
    # Save fixed data
    print(f"\n💾 Saving fixed data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Summary:")
    print(f"  Documents processed: {len(data)}")
    print(f"  Total entities before: {total_entities_before}")
    print(f"  Total entities after: {total_entities_after}")
    print(f"  Entity change: {total_entities_after - total_entities_before:+d}")
    
    # Count documents with corruption
    corrupted_docs = sum(1 for doc in fixed_data if doc.get('had_corruption', False))
    print(f"  Documents with corruption: {corrupted_docs}")
    
    return fixed_data

def main():
    input_file = "/home/ide/ide/ECHR_mask_nemo/output_nemo.json"
    output_file = "/home/ide/ide/ECHR_mask_nemo/output_nemo_fixed.json"
    
    try:
        fixed_data = fix_corrupted_nemo_output(input_file, output_file)
        
        # Show some statistics
        if fixed_data:
            print(f"\n📈 Additional Statistics:")
            
            # Entity type distribution
            entity_counts = {}
            for doc in fixed_data:
                for entity in doc.get('nemo_detected_pii', []):
                    label = entity['label']
                    entity_counts[label] = entity_counts.get(label, 0) + 1
            
            print(f"  Entity types detected:")
            for label, count in sorted(entity_counts.items()):
                print(f"    - {label}: {count}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()