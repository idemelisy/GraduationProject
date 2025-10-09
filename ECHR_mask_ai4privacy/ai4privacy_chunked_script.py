#!/usr/bin/env python3
"""
AI4Privacy chunked processing script

Splits long texts into smaller chunks to overcome context length limitations.
AI4Privacy has a context length of 1536 tokens, so we'll use chunks of ~1000 tokens
with overlaps to ensure entities near boundaries aren't missed.
"""

import json
from ai4privacy import protect
from typing import List, Dict, Tuple
import re
from transformers import AutoTokenizer

class AI4PrivacyChunkedProcessor:
    def __init__(self, chunk_size_tokens=1228, overlap_tokens=153):
        """
        Initialize the chunked processor.
        
        AI4Privacy model max length: 1536 tokens
        Recommended chunk size: 1228 tokens (80% of max)
        Recommended overlap: 153 tokens (10% of max)
        
        Args:
            chunk_size_tokens: Target size for each chunk in tokens
            overlap_tokens: Number of overlapping tokens between chunks
        """
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        
        # Load tokenizer to get exact token counts
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"
            )
            print(f"✅ Loaded tokenizer. Max length: {self.tokenizer.model_max_length}")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("Using character-based estimation (less accurate)")
            self.tokenizer = None
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Fallback: rough estimation (1 token ≈ 4 characters for English)
        return len(text) // 4
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks that respect sentence boundaries.
        
        Returns:
            List of (chunk_text, start_offset) tuples
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        
        current_chunk = ""
        current_start = 0
        current_sentences = []
        
        for sentence in sentences:
            # Find the actual position of this sentence in the original text
            sentence_start = text.find(sentence, current_start)
            if sentence_start == -1:
                # Fallback if exact match fails
                sentence_start = current_start
            
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if current_chunk and self.estimate_tokens(test_chunk) > self.chunk_size_tokens:
                # Save current chunk
                if current_chunk:
                    chunks.append((current_chunk.strip(), current_start))
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                current_chunk = " ".join(overlap_sentences)
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                
                # Update start position (accounting for overlap)
                if overlap_sentences:
                    overlap_text = " ".join(overlap_sentences)
                    current_start = text.find(overlap_text, sentence_start - len(overlap_text) - 100)
                    if current_start == -1:
                        current_start = sentence_start
                else:
                    current_start = sentence_start
                
                current_sentences = overlap_sentences + [sentence]
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = sentence_start
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk:
            chunks.append((current_chunk.strip(), current_start))
        
        return chunks
    
    def process_chunk(self, chunk_text: str, chunk_start: int) -> List[Dict]:
        """Process a single chunk with AI4Privacy."""
        try:
            result = protect(chunk_text, classify_pii=True, verbose=True)
            
            detected_entities = []
            if isinstance(result, dict) and 'replacements' in result:
                for replacement in result['replacements']:
                    # Adjust offsets to be relative to the original document
                    detected_entities.append({
                        'label': replacement['label'],
                        'text': replacement['value'],
                        'start': replacement['start'] + chunk_start,
                        'end': replacement['end'] + chunk_start,
                        'chunk_start': chunk_start,  # For debugging
                        'local_start': replacement['start'],  # Original position in chunk
                        'local_end': replacement['end']
                    })
            
            return detected_entities
        except Exception as e:
            print(f"Error processing chunk starting at {chunk_start}: {e}")
            return []
    
    def merge_overlapping_entities(self, all_entities: List[Dict]) -> List[Dict]:
        """
        Remove duplicate entities that appear in overlapping regions.
        Keep the entity from the chunk where it appears more completely.
        """
        # Sort by start position
        all_entities.sort(key=lambda x: x['start'])
        
        merged = []
        
        for entity in all_entities:
            # Check if this entity overlaps with any already merged entity
            is_duplicate = False
            
            for existing in merged:
                # Check for overlap
                overlap_start = max(entity['start'], existing['start'])
                overlap_end = min(entity['end'], existing['end'])
                
                if overlap_start < overlap_end:
                    # There's an overlap - check if they're the same entity
                    if (entity['text'] == existing['text'] and 
                        entity['label'] == existing['label']):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                merged.append(entity)
        
        return merged
    
    def process_text(self, text: str) -> List[Dict]:
        """Process a long text using chunking strategy."""
        print(f"Processing text of length {len(text)} characters (~{self.estimate_tokens(text)} tokens)")
        
        # Create chunks
        chunks = self.create_chunks(text)
        print(f"Created {len(chunks)} chunks")
        
        # Process each chunk
        all_entities = []
        
        for i, (chunk_text, chunk_start) in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} (start: {chunk_start}, length: {len(chunk_text)})")
            
            chunk_entities = self.process_chunk(chunk_text, chunk_start)
            all_entities.extend(chunk_entities)
            
            print(f"  Found {len(chunk_entities)} entities in this chunk")
        
        # Merge overlapping entities
        print(f"Total entities before merging: {len(all_entities)}")
        merged_entities = self.merge_overlapping_entities(all_entities)
        print(f"Total entities after merging: {len(merged_entities)}")
        
        # Remove debugging fields
        for entity in merged_entities:
            entity.pop('chunk_start', None)
            entity.pop('local_start', None)
            entity.pop('local_end', None)
        
        return merged_entities

def main():
    """Test chunked processing on the first ECHR document."""
    print("=== AI4Privacy Chunked Processing Test ===")
    
    # Load the first document
    with open('/home/ide/ide/data/input.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    first_doc = data[0]
    text = first_doc['text']
    
    print(f"Testing on document: {first_doc['doc_id']}")
    print(f"Text length: {len(text)} characters")
    
    # Create processor with optimized settings
    processor = AI4PrivacyChunkedProcessor(chunk_size_tokens=1228, overlap_tokens=153)
    
    # Process with chunking
    chunked_entities = processor.process_text(text)
    
    # Check results
    print(f"\n=== Results ===")
    print(f"Total entities detected: {len(chunked_entities)}")
    
    # Check specifically for the names we're interested in
    henrik_found = any(entity['text'] == 'Henrik' for entity in chunked_entities)
    hasslund_found = any(entity['text'] == 'Hasslund' for entity in chunked_entities)
    holst_found = any(entity['text'] == 'Holst-Christensen' for entity in chunked_entities)
    
    print(f"Henrik found: {henrik_found}")
    print(f"Hasslund found: {hasslund_found}")
    print(f"Holst-Christensen found: {holst_found}")
    
    # Show first few entities
    print(f"\nFirst 10 entities:")
    for i, entity in enumerate(chunked_entities[:10]):
        print(f"  {i+1}. {entity['label']}: '{entity['text']}' at {entity['start']}-{entity['end']}")
    
    # Save results for comparison
    result = {
        'id': first_doc['doc_id'],
        'original_text': text,
        'text': text,
        'ai4privacy_detected_pii': chunked_entities,
        'processing_method': 'chunked'
    }
    
    with open('/home/ide/ide/ECHR_mask_ai4privacy/output_chunked_test.json', 'w', encoding='utf-8') as f:
        json.dump([result], f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: output_chunked_test.json")

if __name__ == "__main__":
    main()