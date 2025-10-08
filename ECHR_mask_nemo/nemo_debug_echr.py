#!/usr/bin/env python3
"""
nemo_debug_echr.py - Test NEMO with actual ECHR text snippets
"""

import json
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify

def test_echr_snippet():
    """Test NEMO with a snippet from the actual ECHR text that's causing problems"""
    
    # Extract the problematic part from the original text
    test_text = """the summer of 1993.

In 1994, the applicant learnt via a local newspaper that he was the subject of an investigation, as was a private limited stockbrokers company, of which he was part owner.

By letter of 9 June 1994 he informed the police that he was available for an interview, if required. By letter of 14 June 1994 the police confirmed that they were in the process of investigation and informed the applicant that they would talk to him at a later stage.

From November 1994 to September 1995, six discovery orders were issued against two banks, four search warrants were issued and numerous interviews were held."""
    
    print(f"Original text snippet:")
    print(f"'{test_text}'")
    print(f"Length: {len(test_text)}")
    
    # Start Dask
    client = Client()
    print("\nDask client started")
    
    # Configure NEMO
    modifier = PiiModifier(
        language="en",
        supported_entities=["PERSON", "DATE_TIME", "ADDRESS"],
        anonymize_action="replace",
        batch_size=1,
        device="cpu"
    )
    
    modify_pipeline = Modify(modifier)
    
    # Process the text
    df = pd.DataFrame({"text": [test_text]})
    ddf = dd.from_pandas(df, npartitions=1)
    dataset = DocumentDataset(ddf)
    
    try:
        masked_dataset = modify_pipeline(dataset)
        masked_text = masked_dataset.df.compute()["text"].iloc[0]
        
        print(f"\nMasked text:")
        print(f"'{masked_text}'")
        print(f"Masked length: {len(masked_text)}")
        
        # Analyze the output
        import re
        
        # Check for complete tags
        complete_tags = re.findall(r'<[A-Z_]+>', masked_text)
        print(f"\nComplete tags found: {complete_tags}")
        
        # Check for broken tags or corruption
        if "<" in masked_text:
            # Find any incomplete or broken tags
            broken_patterns = [
                r'<[^>]*$',  # Tag starts but doesn't end
                r'^[^<]*>',  # Tag ends but doesn't start
                r'<[^A-Z_>]',  # Tag with invalid characters
                r'[a-zA-Z]<[A-Z_]+>',  # Tag immediately after letter (no space)
                r'<[A-Z_]+>[a-zA-Z]'   # Tag immediately before letter (no space)
            ]
            
            for pattern in broken_patterns:
                matches = re.findall(pattern, masked_text)
                if matches:
                    print(f"❌ Broken pattern '{pattern}': {matches}")
        
        # Compare lengths
        if abs(len(masked_text) - len(test_text)) > 50:  # Allow some difference for tags
            print(f"⚠️  Significant length difference: {len(test_text)} -> {len(masked_text)}")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    client.close()

if __name__ == "__main__":
    test_echr_snippet()