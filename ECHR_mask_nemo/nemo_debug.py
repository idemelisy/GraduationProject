#!/usr/bin/env python3
"""
nemo_debug.py - Debug NEMO PII detection issues
"""

import json
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify

def test_simple_text():
    """Test NEMO with a simple text to see if the issue persists"""
    
    # Simple test text
    test_text = "Mr John Smith was born on January 1, 1990 in New York. He lives at 123 Main Street, Springfield."
    
    print(f"Original text: {test_text}")
    print(f"Length: {len(test_text)}")
    
    # Start Dask
    client = Client()
    print("Dask client started")
    
    # Configure NEMO with minimal settings
    modifier = PiiModifier(
        language="en",
        supported_entities=["PERSON", "DATE_TIME", "ADDRESS"],
        anonymize_action="replace",
        batch_size=1,
        device="cpu"  # Force CPU to avoid GPU issues
    )
    
    modify_pipeline = Modify(modifier)
    
    # Process the text
    df = pd.DataFrame({"text": [test_text]})
    ddf = dd.from_pandas(df, npartitions=1)
    dataset = DocumentDataset(ddf)
    
    try:
        masked_dataset = modify_pipeline(dataset)
        masked_text = masked_dataset.df.compute()["text"].iloc[0]
        
        print(f"\nMasked text: {masked_text}")
        print(f"Masked length: {len(masked_text)}")
        
        # Check for corruption signs
        if "<" in masked_text and ">" in masked_text:
            print("\n✓ Entity tags found")
            # Count complete vs broken tags
            import re
            complete_tags = re.findall(r'<[A-Z_]+>', masked_text)
            broken_tags = re.findall(r'<[^>]*$|^[^<]*>', masked_text)
            print(f"Complete tags: {complete_tags}")
            if broken_tags:
                print(f"❌ Broken tags detected: {broken_tags}")
        else:
            print("❌ No entity tags found!")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
    client.close()

if __name__ == "__main__":
    test_simple_text()