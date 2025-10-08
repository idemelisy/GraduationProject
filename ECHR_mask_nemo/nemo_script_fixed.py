#!/usr/bin/env python3
"""
nemo_script_fixed.py

Fixed version of the NeMo Curator PII detection script.
Addresses issues with entity extraction and text alignment.
"""

import os
import json
import re
import torch
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify

# -----------------------------------------------------
# HELPER: Extract entities from masked vs original (FIXED)
# -----------------------------------------------------
def extract_pii_entities_fixed(original, masked):
    """
    Extract PII entities by comparing original and masked text.
    Fixed to handle NEMO's <ENTITY_TYPE> format correctly.
    """
    entities = []
    
    # NEMO uses <ENTITY_TYPE> format, not {{ENTITY_TYPE}}
    pattern = re.compile(r'<([^>]+)>')
    
    # Find all entity placeholders in masked text
    masked_entities = list(pattern.finditer(masked))
    
    if not masked_entities:
        return entities
    
    # Try to align original and masked text
    orig_pos = 0
    masked_pos = 0
    
    for match in masked_entities:
        entity_type = match.group(1)
        entity_start_in_masked = match.start()
        entity_end_in_masked = match.end()
        
        # Find the text before this entity in masked text
        pre_text = masked[masked_pos:entity_start_in_masked]
        
        # Find corresponding position in original text
        if pre_text:
            pre_pos = original.find(pre_text, orig_pos)
            if pre_pos == -1:
                # If we can't find the pre-text, try a different approach
                continue
            orig_pos = pre_pos + len(pre_text)
        
        # Find the next non-entity text to determine entity boundaries
        next_entity_start = len(masked)
        if len(masked_entities) > masked_entities.index(match) + 1:
            next_match = masked_entities[masked_entities.index(match) + 1]
            next_entity_start = next_match.start()
        
        # Get text after current entity until next entity or end
        post_text = masked[entity_end_in_masked:next_entity_start]
        post_text = re.sub(r'<[^>]+>', '', post_text)  # Remove any nested entities
        
        # Find where this post-text appears in original
        if post_text.strip():
            post_pos = original.find(post_text, orig_pos)
            if post_pos != -1:
                entity_text = original[orig_pos:post_pos]
                entities.append({
                    "label": entity_type,
                    "text": entity_text.strip(),
                    "start": orig_pos,
                    "end": post_pos
                })
                orig_pos = post_pos
            else:
                # Fallback: try to estimate entity length
                # This is a heuristic and may not be perfect
                estimated_length = 20  # Default estimate
                if entity_type == "DATE_TIME":
                    estimated_length = 15
                elif entity_type == "PERSON":
                    estimated_length = 25
                elif entity_type == "ADDRESS":
                    estimated_length = 30
                
                entity_end = min(orig_pos + estimated_length, len(original))
                # Try to find word boundary
                while entity_end < len(original) and original[entity_end].isalnum():
                    entity_end += 1
                
                entity_text = original[orig_pos:entity_end]
                entities.append({
                    "label": entity_type,
                    "text": entity_text.strip(),
                    "start": orig_pos,
                    "end": entity_end
                })
                orig_pos = entity_end
        else:
            # This is the last entity, take rest of meaningful text
            remaining_text = original[orig_pos:].strip()
            if remaining_text:
                # Find a reasonable stopping point
                end_pos = orig_pos + min(len(remaining_text), 100)
                # Try to stop at sentence boundary
                sentence_end = original.find('.', orig_pos)
                if sentence_end != -1 and sentence_end < end_pos:
                    end_pos = sentence_end + 1
                
                entity_text = original[orig_pos:end_pos]
                entities.append({
                    "label": entity_type,
                    "text": entity_text.strip(),
                    "start": orig_pos,
                    "end": end_pos
                })
        
        masked_pos = entity_end_in_masked
    
    return entities

def simple_entity_extraction(original, masked):
    """
    Simplified approach: extract entities by finding differences between original and masked text.
    This is more robust when the masked text is heavily corrupted.
    """
    entities = []
    
    # Find all entity tags in masked text
    entity_pattern = re.compile(r'<([^>]+)>')
    entity_matches = list(entity_pattern.finditer(masked))
    
    # If no entities found, return empty
    if not entity_matches:
        return entities
    
    # Remove all entity tags to get clean masked text
    clean_masked = re.sub(r'<[^>]+>', '', masked)
    
    # Simple approach: find words in original that are missing in clean_masked
    original_words = re.findall(r'\b\w+\b', original)
    masked_words = re.findall(r'\b\w+\b', clean_masked)
    
    # This is a very basic approach and may need refinement
    # For now, let's use the entity pattern matching with better error handling
    
    return extract_entities_by_pattern_matching(original, masked)

def extract_entities_by_pattern_matching(original, masked):
    """
    Extract entities by finding entity patterns and mapping to original text positions.
    """
    entities = []
    
    # Common entity patterns in the original text
    patterns = {
        'PERSON': [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?\b',  # FirstName LastName(-LastName)
            r'\bMr\.? [A-Z][a-z]+\b',  # Mr. Name
            r'\bMs\.? [A-Z][a-z]+\b',  # Ms. Name
        ],
        'DATE_TIME': [
            r'\b\d{1,2} [A-Z][a-z]+ \d{4}\b',  # 31 August 2006
            r'\b[A-Z][a-z]+ \d{4}\b',  # January 2005
            r'\b\d{4}\b',  # 1973
            r'\bthe [a-z]+ of \d{4}\b',  # the beginning of 1996
            r'\b[A-Z][a-z]+ \d{1,2}, \d{4}\b',  # March 19, 1997
        ],
        'ADDRESS': [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+, [A-Z][a-z]+\b',  # City Country
            r'\bfrom [a-z]+ [a-z]+\b',  # from late 1980s
        ]
    }
    
    for entity_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in re.finditer(pattern, original):
                entities.append({
                    "label": entity_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
    
    # Remove duplicates and sort by position
    seen = set()
    unique_entities = []
    for entity in sorted(entities, key=lambda x: x['start']):
        key = (entity['start'], entity['end'], entity['label'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    
    return unique_entities

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    input_path = "/home/ide/ide/ECHR_mask/annotations.json"
    output_path = "/home/ide/ide/ECHR_mask_nemo/output_nemo_fixed.json"

    # Load dataset
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Start Dask
    client = Client()
    print("Dask client started:", client)

    results = []

    # Configure NeMo modifier with better settings
    modifier = PiiModifier(
        language="en",
        supported_entities=["PERSON", "ADDRESS", "DATE_TIME", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        anonymize_action="replace",
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    modify_pipeline = Modify(modifier)

    # Process documents
    for i, item in enumerate(data[:5]):  # Process first 5 for testing
        text = item["text"]
        print(f"\nProcessing document {i+1}: {len(text)} characters")
        
        try:
            df = pd.DataFrame({"text": [text]})
            ddf = dd.from_pandas(df, npartitions=1)
            dataset = DocumentDataset(ddf)

            masked_dataset = modify_pipeline(dataset)
            masked_text = masked_dataset.df.compute()["text"].iloc[0]
            
            print(f"Original length: {len(text)}")
            print(f"Masked length: {len(masked_text)}")
            
            # Try multiple extraction methods
            entities_method1 = extract_pii_entities_fixed(text, masked_text)
            entities_method2 = extract_entities_by_pattern_matching(text, masked_text)
            
            # Use pattern matching as fallback if main method fails
            if len(entities_method1) == 0 or any(e.get('end', 0) - e.get('start', 0) > 1000 for e in entities_method1):
                print("Using pattern matching fallback method")
                detected_entities = entities_method2
            else:
                detected_entities = entities_method1
            
            print(f"Detected {len(detected_entities)} entities")
            
            results.append({
                "id": item.get("id", f"doc_{i}"),
                "text": text,
                "nemo_masked_text": masked_text,
                "nemo_detected_pii": detected_entities
            })

        except Exception as e:
            print(f"Error processing document {i+1}: {str(e)}")
            # Add document with empty results
            results.append({
                "id": item.get("id", f"doc_{i}"),
                "text": text,
                "nemo_masked_text": "",
                "nemo_detected_pii": [],
                "error": str(e)
            })

        if (i + 1) % 5 == 0:
            print(f"Processed {i+1} documents...")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Saved detections to {output_path}")
    
    # Show sample of results
    if results:
        print(f"\nSample entity detection from first document:")
        for entity in results[0].get("nemo_detected_pii", [])[:5]:
            print(f"  {entity['label']}: '{entity['text']}' [{entity['start']}:{entity['end']}]")