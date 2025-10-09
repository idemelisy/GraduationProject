#!/usr/bin/env python3
"""
Check for spacing issues around Henrik Hasslund in the original text
"""

import json

def check_spacing_issues():
    """Check the exact characters around Henrik Hasslund."""
    
    # Load the original data
    with open('/home/ide/ide/data/input.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get the first document
    first_doc = data[0]
    text = first_doc['text']
    
    # Find Henrik Hasslund
    henrik_pos = text.find("Henrik Hasslund")
    if henrik_pos != -1:
        # Show characters around it
        start = max(0, henrik_pos - 20)
        end = min(len(text), henrik_pos + 35)
        context = text[start:end]
        
        print("=== Context around 'Henrik Hasslund' ===")
        print(f"Position: {henrik_pos}")
        print(f"Context: '{context}'")
        print()
        
        # Show each character with its ASCII code
        print("Character analysis:")
        for i, char in enumerate(context):
            pos = start + i
            if henrik_pos <= pos <= henrik_pos + 14:  # Henrik Hasslund is 14 chars
                marker = " <-- HERE"
            else:
                marker = ""
            print(f"  {pos:3d}: '{char}' (ASCII: {ord(char)}){marker}")
    else:
        print("Henrik Hasslund not found!")
    
    # Also check the AI4Privacy output
    print("\n=== Checking AI4Privacy output ===")
    with open('/home/ide/ide/ECHR_mask_ai4privacy/output_ai4privacy.json', 'r', encoding='utf-8') as f:
        ai4_data = json.load(f)
    
    first_result = ai4_data[0]
    ai4_text = first_result['text']
    
    # Check if the text is the same
    print(f"Original text length: {len(text)}")
    print(f"AI4Privacy text length: {len(ai4_text)}")
    print(f"Texts are identical: {text == ai4_text}")
    
    if text != ai4_text:
        print("Texts differ! This could be the issue.")
        # Find the difference
        for i, (c1, c2) in enumerate(zip(text, ai4_text)):
            if c1 != c2:
                print(f"First difference at position {i}: original='{c1}' vs ai4privacy='{c2}'")
                break

if __name__ == "__main__":
    check_spacing_issues()