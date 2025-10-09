#!/usr/bin/env python3
"""
Test with the exact sentence from the ECHR document
"""

from ai4privacy import protect

def test_exact_sentence():
    """Test the exact sentence from the ECHR document."""
    
    exact_sentence = 'PROCEDURE The case originated in an application (no. 36244/06) against the Kingdom of Denmark lodged with the Court under Article 34 of the Convention for the Protection of Human Rights and Fundamental Freedoms ("the Convention") by a Danish national, Mr Henrik Hasslund ("the applicant"), on 31 August 2006.'
    
    print("=== Testing exact sentence from ECHR document ===")
    print(f"Text: {exact_sentence}")
    print()
    
    try:
        result = protect(exact_sentence, classify_pii=True, verbose=True)
        
        if isinstance(result, dict) and 'replacements' in result:
            if result['replacements']:
                print("Detected entities:")
                for replacement in result['replacements']:
                    print(f"  - {replacement['label']}: '{replacement['value']}' at position {replacement['start']}-{replacement['end']}")
                    
                # Check specifically for Hasslund
                hasslund_found = any(replacement['value'] == 'Hasslund' for replacement in result['replacements'])
                print(f"\nWas 'Hasslund' detected? {hasslund_found}")
            else:
                print("No entities detected")
                
        else:
            print(f"Unexpected result format: {result}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_exact_sentence()