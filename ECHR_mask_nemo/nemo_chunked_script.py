#!/usr/bin/env python3
"""
nemo_chunked_script.py

Process ECHR texts using NEMO with chunking to avoid corruption issues.
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

def chunk_text(text, max_chunk_size=2000, overlap=200):
    """
    Split text into overlapping chunks to avoid processing issues.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks to avoid splitting entities
    
    Returns:
        List of (chunk_text, start_offset) tuples
    """
    if len(text) <= max_chunk_size:
        return [(text, 0)]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        
        # Try to end at a sentence boundary if possible
        if end < len(text):
            # Look for sentence endings within the last 200 characters
            sentence_end = -1
            for i in range(end - 200, end):
                if i > 0 and text[i-1] in '.!?' and text[i].isspace():
                    sentence_end = i
            
            if sentence_end > start + max_chunk_size // 2:
                end = sentence_end
        
        chunk = text[start:end]
        chunks.append((chunk, start))
        
        if end >= len(text):
            break
            
        # Move start forward, but with overlap
        start = end - overlap
    
    return chunks

def process_chunk_with_nemo(chunk_text, modifier, modify_pipeline):
    """
    Process a single chunk with NEMO, with error handling.
    
    Returns:
        (masked_text, success) tuple
    """
    try:
        df = pd.DataFrame({"text": [chunk_text]})
        ddf = dd.from_pandas(df, npartitions=1)
        dataset = DocumentDataset(ddf)

        masked_dataset = modify_pipeline(dataset)
        masked_text = masked_dataset.df.compute()["text"].iloc[0]
        
        # Check for obvious corruption
        if len(masked_text) > len(chunk_text) * 2:
            print(f"⚠️  Warning: Chunk output size suspicious ({len(chunk_text)} -> {len(masked_text)})")
            return chunk_text, False  # Return original if corrupted
        
        # Check for broken tags
        if re.search(r'[a-zA-Z]<[A-Z_]+>|<[A-Z_]+>[a-zA-Z]', masked_text):
            print(f"⚠️  Warning: Broken tags detected in chunk")
            return chunk_text, False  # Return original if corrupted
            
        return masked_text, True
        
    except Exception as e:
        print(f"❌ Error processing chunk: {e}")
        return chunk_text, False

def merge_chunked_results(chunks_with_offsets, masked_chunks):
    """
    Merge masked chunks back into a single text, handling overlaps.
    
    Args:
        chunks_with_offsets: Original chunks with their start positions
        masked_chunks: Corresponding masked chunks
    
    Returns:
        Merged masked text
    """
    if len(chunks_with_offsets) == 1:
        return masked_chunks[0]
    
    # For simplicity, just concatenate non-overlapping parts
    # In a more sophisticated version, we'd merge the overlapping regions
    merged_text = masked_chunks[0]
    
    for i in range(1, len(masked_chunks)):
        # Add the non-overlapping part of the next chunk
        prev_chunk, prev_offset = chunks_with_offsets[i-1]
        curr_chunk, curr_offset = chunks_with_offsets[i]
        
        # Calculate where the overlap starts in the current chunk
        overlap_start = prev_offset + len(prev_chunk) - curr_offset
        if overlap_start > 0 and overlap_start < len(masked_chunks[i]):
            # Add only the non-overlapping part
            merged_text += masked_chunks[i][overlap_start:]
        else:
            # No overlap or calculation error, just add the whole chunk
            merged_text += " " + masked_chunks[i]
    
    return merged_text

def extract_entities_from_chunked_text(original_text, merged_masked_text):
    """
    Extract PII entities by comparing original and merged masked text.
    """
    entities = []
    
    # Find all entity tags in the masked text
    entity_pattern = re.compile(r'<([A-Z_]+)>')
    
    # Split both texts into words for alignment
    original_words = re.findall(r'\S+', original_text)
    masked_words = re.findall(r'<[A-Z_]+>|\S+', merged_masked_text)
    
    orig_pos = 0
    orig_word_idx = 0
    
    for masked_word in masked_words:
        if entity_pattern.match(masked_word):
            # This is an entity tag
            entity_type = entity_pattern.match(masked_word).group(1)
            
            # Find the corresponding word(s) in the original text
            if orig_word_idx < len(original_words):
                orig_word = original_words[orig_word_idx]
                
                # Find this word's position in the original text
                word_start = original_text.find(orig_word, orig_pos)
                if word_start != -1:
                    word_end = word_start + len(orig_word)
                    
                    entities.append({
                        "label": entity_type,
                        "text": orig_word,
                        "start": word_start,
                        "end": word_end
                    })
                    
                    orig_pos = word_end
                    orig_word_idx += 1
        else:
            # This is a regular word, advance in original text
            if orig_word_idx < len(original_words):
                orig_word = original_words[orig_word_idx]
                orig_pos = original_text.find(orig_word, orig_pos) + len(orig_word)
                orig_word_idx += 1
    
    return entities

def main():
    input_path = "/home/ide/ide/ECHR_mask/annotations.json"
    output_path = "/home/ide/ide/ECHR_mask_nemo/output_nemo_chunked.json"

    # Load dataset
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} documents with chunking approach...")

    # Start Dask
    client = Client()
    print("Dask client started")

    # Configure NeMo modifier
    print("Initializing NEMO PII modifier...")
    modifier = PiiModifier(
        language="en",
        supported_entities=["PERSON", "ADDRESS", "DATE_TIME", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        anonymize_action="replace",
        batch_size=1,
        device="cpu"  # Use CPU to avoid GPU memory issues
    )

    modify_pipeline = Modify(modifier)
    print("NEMO initialization complete")

    results = []

    # Process documents
    for i, item in enumerate(data[:3]):  # Process first 3 for testing
        text = item["text"]
        print(f"\nProcessing document {i+1}/{len(data[:3])}: {len(text)} characters")
        
        try:
            # Chunk the text
            chunks = chunk_text(text, max_chunk_size=1500, overlap=150)
            print(f"  Split into {len(chunks)} chunks")
            
            # Process each chunk
            masked_chunks = []
            all_successful = True
            
            for j, chunk_data in enumerate(chunks):
                chunk_text, offset = chunk_data
                print(f"    Processing chunk {j+1}/{len(chunks)} (size: {len(chunk_text)})")
                masked_chunk, success = process_chunk_with_nemo(chunk_text, modifier, modify_pipeline)
                masked_chunks.append(masked_chunk)
                
                if not success:
                    all_successful = False
                    print(f"    ⚠️  Chunk {j+1} processing failed, using original text")
            
            # Merge chunks back together
            if all_successful:
                merged_masked_text = merge_chunked_results(chunks, masked_chunks)
                print(f"  ✓ Successfully merged chunks: {len(text)} -> {len(merged_masked_text)} chars")
                
                # Extract entities
                detected_entities = extract_entities_from_chunked_text(text, merged_masked_text)
                print(f"  ✓ Detected {len(detected_entities)} entities")
                
            else:
                # Fallback: use original text if too many chunks failed
                merged_masked_text = text
                detected_entities = []
                print(f"  ❌ Too many chunk failures, using original text")

            results.append({
                "id": item.get("id", f"doc_{i}"),
                "text": text,
                "nemo_masked_text": merged_masked_text,
                "nemo_detected_pii": detected_entities,
                "processing_chunks": len(chunks),
                "processing_successful": all_successful
            })

        except Exception as e:
            print(f"❌ Error processing document {i+1}: {str(e)}")
            results.append({
                "id": item.get("id", f"doc_{i}"),
                "text": text,
                "nemo_masked_text": text,
                "nemo_detected_pii": [],
                "error": str(e)
            })

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Saved results to {output_path}")
    
    # Show summary
    if results:
        total_entities = sum(len(r.get("nemo_detected_pii", [])) for r in results)
        successful_docs = sum(1 for r in results if r.get("processing_successful", False))
        print(f"\nSummary:")
        print(f"  Documents processed: {len(results)}")
        print(f"  Successful processing: {successful_docs}/{len(results)}")
        print(f"  Total entities detected: {total_entities}")
        
        # Show sample entities
        for i, result in enumerate(results[:2]):
            entities = result.get("nemo_detected_pii", [])
            if entities:
                print(f"\nSample entities from document {i+1}:")
                for entity in entities[:5]:
                    print(f"  {entity['label']}: '{entity['text']}' [{entity['start']}:{entity['end']}]")

    client.close()

if __name__ == "__main__":
    main()