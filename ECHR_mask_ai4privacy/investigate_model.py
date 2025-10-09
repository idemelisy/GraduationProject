#!/usr/bin/env python3
"""
Investigate AI4Privacy model specifications
"""

import json
from ai4privacy import protect
import transformers
import torch

def investigate_model_specs():
    """Try to find AI4Privacy model specifications."""
    
    print("=== AI4Privacy Model Investigation ===")
    
    # Try to get model information
    try:
        # First, let's see what happens when we call protect with a very simple text
        # This will force the model to load and we can see what model it uses
        simple_result = protect("Test text", classify_pii=True, verbose=True)
        print("AI4Privacy loaded successfully")
        
        # Try to find the model configuration
        # AI4Privacy typically uses transformers models
        print("\nTrying to access model information...")
        
        # Check if we can import and inspect the model directly
        try:
            from ai4privacy.anonymizer import PiiAnonymizer
            print("Found PiiAnonymizer class")
        except ImportError:
            print("Could not import PiiAnonymizer directly")
        
        # Based on the loading message, it uses: ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii
        model_name = "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"
        print(f"Model name: {model_name}")
        
        # Try to get tokenizer info
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if hasattr(tokenizer, 'model_max_length'):
                print(f"Model max length: {tokenizer.model_max_length}")
            
            if hasattr(tokenizer, 'max_len'):
                print(f"Max length: {tokenizer.max_len}")
                
        except Exception as e:
            print(f"Could not load tokenizer: {e}")
            
        # Test with different text lengths to find the breaking point
        print("\n=== Testing different text lengths ===")
        
        test_lengths = [100, 500, 1000, 2000, 4000, 8000]
        
        for length in test_lengths:
            test_text = "This is a test sentence with a name John Smith. " * (length // 50)
            test_text = test_text[:length]
            
            try:
                result = protect(test_text, classify_pii=True, verbose=True)
                entities_found = len(result.get('replacements', []))
                print(f"Length {length:4d}: Success - {entities_found} entities found")
            except Exception as e:
                print(f"Length {length:4d}: Failed - {e}")
                break
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    investigate_model_specs()