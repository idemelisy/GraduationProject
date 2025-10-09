#!/usr/bin/env python3
"""
Test script to understand why AI4Privacy detects some names but not others
"""

from ai4privacy import protect

def test_name_detection():
    """Test AI4Privacy detection on specific names."""
    
    # Test sentences
    test_sentences = [
        "The applicant was Mr Henrik Hasslund from Denmark.",
        "The agent was Ms Nina Holst-Christensen from the Ministry.",
        "Mr Hasslund was represented by a lawyer.",
        "Ms Holst-Christensen attended the meeting.",
        "Henrik Hasslund is a Danish citizen.",
        "Nina Holst-Christensen works for the government.",
        "The case involves Mr Henrik Hasslund versus Ms Nina Holst-Christensen.",
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n=== Test {i}: {sentence} ===")
        
        try:
            result = protect(sentence, classify_pii=True, verbose=True)
            
            if isinstance(result, dict) and 'replacements' in result:
                if result['replacements']:
                    print("Detected entities:")
                    for replacement in result['replacements']:
                        print(f"  - {replacement['label']}: '{replacement['value']}' at position {replacement['start']}-{replacement['end']}")
                else:
                    print("No entities detected")
            else:
                print(f"Unexpected result format: {result}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_name_detection()