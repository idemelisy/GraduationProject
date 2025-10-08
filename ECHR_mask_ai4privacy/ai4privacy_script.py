#!/usr/bin/env python3
"""
ai4privacy_script.py

AI4Privacy processing script that:
- Loads annotations.json with ground truth in ECHR format  
- Runs AI4Privacy PII detection with categorical labels using protect()
- Saves raw results for later evaluation with post-processing
"""

import json
from ai4privacy import protect
from typing import List, Dict
import re

def process_single_text_with_ai4privacy(text: str) -> List[Dict]:
    """
    Process a single text with AI4Privacy and return detected entities.
    Uses protect() function with classify_pii=True and verbose=True to get categorical labels.
    """
    if not text.strip():
        return []
    
    try:
        # Use AI4Privacy protect function with categorical classification
        result = protect(text, classify_pii=True, verbose=True)
        
        # Extract entities from the verbose result
        detected_entities = []
        if isinstance(result, dict) and 'replacements' in result:
            for replacement in result['replacements']:
                detected_entities.append({
                    'label': replacement['label'],
                    'text': replacement['value'],
                    'start': replacement['start'],
                    'end': replacement['end']
                })
        
        return detected_entities
        
    except Exception as e:
        print(f"Error processing text with AI4Privacy: {e}")
        return []

def main():
    """Main processing function - loads ECHR data and processes with AI4Privacy."""
    print("=== AI4Privacy PII Detection Script ===")
    
    try:
        # Load annotations
        print("Loading annotations...")
        with open('/home/ide/ide/ECHR_mask/annotations.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} documents")
        
        # Process each document
        results = []
        
        for i, doc in enumerate(data):
            if i % 50 == 0:
                print(f"Processing document {i+1}/{len(data)}")
            
            text = doc['text']
            doc_id = doc['doc_id']
            
            # Get ground truth entities
            gt_entities = []
            for annotation in doc.get('annotations', []):
                gt_entities.append({
                    'entity_type': annotation['entity_type'],
                    'start_offset': annotation['start_offset'],
                    'end_offset': annotation['end_offset'],
                    'span_text': annotation['span_text']
                })
            
            # Process with AI4Privacy
            detected_entities = process_single_text_with_ai4privacy(text)
            
            # Store raw results
            result_doc = {
                'id': doc_id,
                'original_text': text,
                'text': text,  # Keep original format
                'ai4privacy_detected_pii': detected_entities,
                'ground_truth_entities': gt_entities
            }
            
            results.append(result_doc)
        
        # Save results
        output_path = '/home/ide/ide/ECHR_mask_ai4privacy/output_ai4privacy.json'
        print(f"Saving {len(results)} results to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("✅ AI4Privacy processing complete!")
        print(f"Results saved to: {output_path}")
        print("Run ai4privacy_evaluate.py to analyze performance")
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()