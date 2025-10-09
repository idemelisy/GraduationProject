#!/usr/bin/env python3
"""
Test AI4Privacy consistency on the same text multiple times
"""

import json
from ai4privacy import protect

def test_consistency():
    """Test if AI4Privacy gives consistent results on the same text."""
    
    # Load the exact text from the first document
    with open('/home/ide/ide/data/input.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = data[0]['text']
    print(f"Testing AI4Privacy consistency on text length: {len(text)}")
    print(f"Text starts with: {text[:100]}...")
    
    # Run AI4Privacy multiple times
    results = []
    
    for run in range(3):
        print(f"\n=== Run {run + 1} ===")
        
        try:
            result = protect(text, classify_pii=True, verbose=True)
            
            detected_entities = []
            if isinstance(result, dict) and 'replacements' in result:
                for replacement in result['replacements']:
                    detected_entities.append({
                        'label': replacement['label'],
                        'text': replacement['value'],
                        'start': replacement['start'],
                        'end': replacement['end']
                    })
            
            results.append(detected_entities)
            
            # Check specifically for Henrik and Hasslund
            henrik_found = any(entity['text'] == 'Henrik' for entity in detected_entities)
            hasslund_found = any(entity['text'] == 'Hasslund' for entity in detected_entities)
            holst_found = any(entity['text'] == 'Holst-Christensen' for entity in detected_entities)
            
            print(f"Henrik found: {henrik_found}")
            print(f"Hasslund found: {hasslund_found}")
            print(f"Holst-Christensen found: {holst_found}")
            print(f"Total entities detected: {len(detected_entities)}")
            
        except Exception as e:
            print(f"Error in run {run + 1}: {e}")
    
    # Compare results
    print(f"\n=== Consistency Analysis ===")
    if len(results) >= 2:
        same_count = len(results[0])
        all_same = True
        for i in range(1, len(results)):
            if len(results[i]) != len(results[0]):
                all_same = False
                print(f"Run {i+1} detected {len(results[i])} entities vs Run 1's {len(results[0])}")
        
        if all_same:
            print("All runs detected the same number of entities")
        else:
            print("Inconsistent results between runs!")

if __name__ == "__main__":
    test_consistency()