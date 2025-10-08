#!/usr/bin/env python3
"""
nemo_simple_chunked.py - Simplified chunked processing with better error handling
"""

import json
import re
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify

def simple_chunk_text(text, chunk_size=1500):
    """Simple chunking without overlap for now"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append((chunk, i))
    return chunks

def process_chunk_safely(chunk_text, modify_pipeline):
    """Process a chunk with comprehensive error handling"""
    try:
        print(f"      Processing chunk of {len(chunk_text)} characters...")
        
        # Create dataframe
        df = pd.DataFrame({"text": [chunk_text]})
        ddf = dd.from_pandas(df, npartitions=1)
        dataset = DocumentDataset(ddf)

        # Process with NEMO
        masked_dataset = modify_pipeline(dataset)
        masked_text = masked_dataset.df.compute()["text"].iloc[0]
        
        # Basic sanity checks
        if not isinstance(masked_text, str):
            print(f"      ❌ Output is not a string: {type(masked_text)}")
            return chunk_text, False
            
        if len(masked_text) > len(chunk_text) * 3:
            print(f"      ❌ Output too long: {len(chunk_text)} -> {len(masked_text)}")
            return chunk_text, False
        
        print(f"      ✓ Chunk processed successfully: {len(chunk_text)} -> {len(masked_text)}")
        return masked_text, True
        
    except Exception as e:
        print(f"      ❌ Error in chunk processing: {str(e)}")
        return chunk_text, False

def extract_simple_entities(original_text, masked_text):
    """Simple entity extraction"""
    entities = []
    
    # Find entity tags
    entity_pattern = re.compile(r'<([A-Z_]+)>')
    
    # For now, just find the positions of tags and map back to original
    # This is a simplified approach
    for match in entity_pattern.finditer(masked_text):
        entity_type = match.group(1)
        # Try to find corresponding text in original (simplified)
        # In a real implementation, we'd do proper alignment
        entities.append({
            "label": entity_type,
            "text": f"[{entity_type}_DETECTED]",
            "start": match.start(),
            "end": match.end()
        })
    
    return entities

def main():
    input_path = "/home/ide/ide/ECHR_mask/annotations.json"
    output_path = "/home/ide/ide/ECHR_mask_nemo/output_nemo_simple_chunked.json"

    # Load data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} documents (testing first 2)...")

    # Initialize Dask
    client = Client()
    print("✓ Dask client started")

    # Initialize NEMO
    try:
        print("Initializing NEMO...")
        modifier = PiiModifier(
            language="en",
            supported_entities=["PERSON", "DATE_TIME", "ADDRESS"],
            anonymize_action="replace",
            batch_size=1,
            device="cpu"
        )
        modify_pipeline = Modify(modifier)
        print("✓ NEMO initialized successfully")
    except Exception as e:
        print(f"❌ NEMO initialization failed: {e}")
        return

    results = []

    # Process documents
    for i, item in enumerate(data[:2]):  # Test with first 2 documents
        text = item["text"]
        doc_id = item.get("id", f"doc_{i}")
        
        print(f"\n📄 Processing document {i+1}: '{doc_id}' ({len(text)} chars)")
        
        try:
            # Chunk the text
            chunks = simple_chunk_text(text, chunk_size=1000)
            print(f"  📋 Split into {len(chunks)} chunks")
            
            # Process chunks
            masked_chunks = []
            successful_chunks = 0
            
            for j, (chunk_text, offset) in enumerate(chunks):
                print(f"  🔄 Chunk {j+1}/{len(chunks)}:")
                
                masked_chunk, success = process_chunk_safely(chunk_text, modify_pipeline)
                masked_chunks.append(masked_chunk)
                
                if success:
                    successful_chunks += 1
            
            print(f"  📊 Successful chunks: {successful_chunks}/{len(chunks)}")
            
            # Merge results
            merged_text = "".join(masked_chunks)
            entities = extract_simple_entities(text, merged_text)
            
            results.append({
                "id": doc_id,
                "text": text,
                "nemo_masked_text": merged_text,
                "nemo_detected_pii": entities,
                "chunks_processed": len(chunks),
                "chunks_successful": successful_chunks,
                "processing_success_rate": successful_chunks / len(chunks) if chunks else 0
            })
            
            print(f"  ✅ Document completed: {len(entities)} entities detected")
            
        except Exception as e:
            print(f"  ❌ Document failed: {str(e)}")
            results.append({
                "id": doc_id,
                "text": text,
                "nemo_masked_text": text,
                "nemo_detected_pii": [],
                "error": str(e)
            })

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Processing complete! Results saved to {output_path}")
    
    # Summary
    if results:
        total_entities = sum(len(r.get("nemo_detected_pii", [])) for r in results)
        successful_docs = sum(1 for r in results if "error" not in r)
        
        print(f"\n📈 Summary:")
        print(f"  Documents processed: {len(results)}")
        print(f"  Successful: {successful_docs}")
        print(f"  Total entities: {total_entities}")

    client.close()

if __name__ == "__main__":
    main()