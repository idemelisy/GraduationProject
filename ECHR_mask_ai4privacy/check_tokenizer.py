#!/usr/bin/env python3
"""
Check AI4Privacy tokenizer specifications directly
"""

from transformers import AutoTokenizer

def check_tokenizer_specs():
    """Check the AI4Privacy tokenizer specifications."""
    
    print("=== Checking AI4Privacy Tokenizer Specs ===")
    
    # Try different model names that might be used by AI4Privacy
    model_names = [
        "AI4Privacy/PII-detection",
        "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii",
        "microsoft/DialoGPT-medium",  # Common fallback
    ]
    
    for model_name in model_names:
        print(f"\n--- Trying model: {model_name} ---")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"✅ Successfully loaded tokenizer for: {model_name}")
            print(f"Model max length: {tokenizer.model_max_length}")
            
            if hasattr(tokenizer, 'max_len'):
                print(f"Max length: {tokenizer.max_len}")
            
            if hasattr(tokenizer, 'max_position_embeddings'):
                print(f"Max position embeddings: {tokenizer.max_position_embeddings}")
                
            # Test tokenization of our problematic text
            test_text = "PROCEDURE The case originated in an application (no. 36244/06) against the Kingdom of Denmark lodged with the Court under Article 34 of the Convention for the Protection of Human Rights and Fundamental Freedoms (\"the Convention\") by a Danish national, Mr Henrik Hasslund (\"the applicant\"), on 31 August 2006."
            
            tokens = tokenizer.encode(test_text)
            print(f"Test sentence tokens: {len(tokens)}")
            print(f"Sample tokens: {tokens[:10]}...")
            
            # Check our full document length
            with open('/home/ide/ide/data/input.json', 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
                full_text = data[0]['text']
                
                full_tokens = tokenizer.encode(full_text)
                print(f"Full document tokens: {len(full_tokens)}")
                print(f"Exceeds max length by: {len(full_tokens) - tokenizer.model_max_length} tokens")
                
            break  # If successful, stop trying other models
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
    
    print("\n=== Token Length Analysis ===")
    # Calculate recommended chunk sizes
    if 'tokenizer' in locals():
        max_length = tokenizer.model_max_length
        recommended_chunk = int(max_length * 0.8)  # 80% of max to be safe
        overlap = int(max_length * 0.1)  # 10% overlap
        
        print(f"Model max length: {max_length} tokens")
        print(f"Recommended chunk size: {recommended_chunk} tokens")
        print(f"Recommended overlap: {overlap} tokens")

if __name__ == "__main__":
    check_tokenizer_specs()