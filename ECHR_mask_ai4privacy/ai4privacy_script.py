#!/usr/bin/env python3
"""
ai4privacy_script.py

AI4Privacy processing script that:
- Loads annotations.json with ground truth in ECHR format  
- Runs AI4Privacy PII detection with categorical labels using protect()
- Uses chunking to handle long documents that exceed AI4Privacy's 1536 token limit
- Saves raw results for later evaluation with post-processing
"""

import json
from ai4privacy import protect
from typing import List, Dict, Tuple
import re

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: transformers not available, using character-based token estimation")

class AI4PrivacyChunkedProcessor:
    """
    Chunked processor for AI4Privacy to handle long documents.
    AI4Privacy has a 1536 token limit, so we split long texts into chunks.
    """
    
    def __init__(self):
        """Initialize the chunked processor."""
        self.max_tokens = 1536
        self.chunk_size_tokens = 1200  # Safe margin
        self.overlap_tokens = 150      # Overlap to catch boundary entities
        
        # Load tokenizer for accurate token counting
        self.tokenizer = None
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"
                )
            except Exception:
                pass
    
    def count_tokens(self, text: str) -> int:
        """Count actual tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except:
                pass
        # Fallback estimation (1 token ≈ 4 characters)
        return len(text) // 4
    
    def split_into_sentences(self, text: str) -> List[Tuple[str, int]]:
        """Split text into sentences with their positions."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = []
        
        current_pos = 0
        for match in re.finditer(sentence_pattern, text):
            sentence = text[current_pos:match.start()].strip()
            if sentence:
                sentences.append((sentence, current_pos))
            current_pos = match.end()
        
        # Add final sentence
        if current_pos < len(text):
            final_sentence = text[current_pos:].strip()
            if final_sentence:
                sentences.append((final_sentence, current_pos))
        
        return sentences
    
    def create_chunks(self, text: str) -> List[Tuple[str, int]]:
        """
        Create chunks that respect token limits and sentence boundaries.
        
        Returns:
            List of (chunk_text, start_pos) tuples
        """
        total_tokens = self.count_tokens(text)
        
        if total_tokens <= self.chunk_size_tokens:
            return [(text, 0)]
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        chunk_start_pos = 0
        
        for i, (sentence, sentence_pos) in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_chunk_tokens + sentence_tokens > self.chunk_size_tokens and current_chunk_sentences:
                # Save current chunk
                chunk_text = ' '.join([s[0] for s in current_chunk_sentences])
                chunks.append((chunk_text, chunk_start_pos))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_tokens = 0
                
                # Add sentences from the end of current chunk for overlap
                for j in range(len(current_chunk_sentences) - 1, -1, -1):
                    sent_text, sent_pos = current_chunk_sentences[j]
                    sent_tokens = self.count_tokens(sent_text)
                    
                    if overlap_tokens + sent_tokens <= self.overlap_tokens:
                        overlap_sentences.insert(0, (sent_text, sent_pos))
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                # Start new chunk
                current_chunk_sentences = overlap_sentences + [(sentence, sentence_pos)]
                current_chunk_tokens = overlap_tokens + sentence_tokens
                chunk_start_pos = overlap_sentences[0][1] if overlap_sentences else sentence_pos
                
            else:
                # Add sentence to current chunk
                if not current_chunk_sentences:
                    chunk_start_pos = sentence_pos
                current_chunk_sentences.append((sentence, sentence_pos))
                current_chunk_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join([s[0] for s in current_chunk_sentences])
            chunks.append((chunk_text, chunk_start_pos))
        
        return chunks
    
    def process_chunk(self, chunk_text: str, chunk_start: int) -> List[Dict]:
        """Process a single chunk with AI4Privacy."""
        try:
            result = protect(chunk_text, classify_pii=True, verbose=True)
            
            entities = []
            if isinstance(result, dict) and 'replacements' in result:
                for replacement in result['replacements']:
                    entities.append({
                        'label': replacement['label'],
                        'text': replacement['value'],
                        'start': replacement['start'] + chunk_start,
                        'end': replacement['end'] + chunk_start
                    })
            
            return entities
            
        except Exception as e:
            print(f"Error processing chunk at position {chunk_start}: {e}")
            return []
    
    def remove_duplicates(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities from overlapping chunks."""
        entities.sort(key=lambda x: (x['start'], x['end']))
        
        unique_entities = []
        
        for entity in entities:
            # Check if this entity overlaps with any existing one
            is_duplicate = False
            
            for existing in unique_entities:
                # Check for position overlap and same text/label
                if (entity['text'] == existing['text'] and 
                    entity['label'] == existing['label'] and
                    abs(entity['start'] - existing['start']) <= 5):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities
    
    def process_text(self, text: str) -> List[Dict]:
        """Process text using chunking if necessary."""
        total_tokens = self.count_tokens(text)
        
        # Use chunking for texts longer than our safe limit
        if total_tokens > self.chunk_size_tokens:
            chunks = self.create_chunks(text)
            
            all_entities = []
            for chunk_text, chunk_start in chunks:
                chunk_entities = self.process_chunk(chunk_text, chunk_start)
                all_entities.extend(chunk_entities)
            
            # Remove duplicates from overlapping chunks
            return self.remove_duplicates(all_entities)
        else:
            # Process normally for short texts
            return self.process_chunk(text, 0)

# Global processor instance to avoid reloading tokenizer/model
_processor_instance = None

def get_processor():
    """Get a global processor instance to avoid reloading tokenizer."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = AI4PrivacyChunkedProcessor()
    return _processor_instance

def process_single_text_with_ai4privacy(text: str) -> List[Dict]:
    """
    Process a single text with AI4Privacy using chunking if necessary.
    Uses protect() function with classify_pii=True and verbose=True to get categorical labels.
    """
    if not text.strip():
        return []
    
    try:
        # Use global processor instance
        processor = get_processor()
        
        # Process with chunking if necessary
        detected_entities = processor.process_text(text)
        
        return detected_entities
        
    except Exception as e:
        print(f"Error processing text with AI4Privacy: {e}")
        return []

def main():
    """Main processing function - loads ECHR data and processes with AI4Privacy using chunking."""
    print("=== AI4Privacy PII Detection Script (with Chunking) ===")
    
    try:
        # Load input
        print("Loading the input...")
        with open('/home/ide/ide/data/input.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} documents")
        
        # Process each document
        results = []
        chunked_docs = 0
        
        for i, doc in enumerate(data):
            if i % 50 == 0:
                print(f"Processing document {i+1}/{len(data)}")
            
            text = doc['text']
            doc_id = doc['doc_id']
            
            # Check if document will need chunking
            processor = get_processor()
            text_tokens = processor.count_tokens(text)
            needs_chunking = text_tokens > processor.chunk_size_tokens
            
            if needs_chunking:
                chunked_docs += 1
                if i < 10:  # Show details for first few chunked documents
                    print(f"  Document {doc_id}: {len(text)} chars, ~{text_tokens} tokens - using chunking")
            
            # Get annotations from input data
            annotations = doc.get('annotations', [])
            
            # Get ground truth entities
            gt_entities = []
            for annotation in annotations:
                gt_entities.append({
                    'entity_type': annotation['entity_type'],
                    'start_offset': annotation['start_offset'],
                    'end_offset': annotation['end_offset'],
                    'span_text': annotation['span_text'],
                    'annotator': annotation.get('annotator', 'unknown')
                })
            
            # Process with AI4Privacy (with chunking if needed)
            detected_entities = process_single_text_with_ai4privacy(text)
            
            # Store raw results
            result_doc = {
                'id': doc_id,
                'text': text,  # Using single text field since no modification is made
                'annotations': annotations,  # Include original annotations
                'ai4privacy_detected_pii': detected_entities,
                'ground_truth_entities': gt_entities,
                'processing_info': {
                    'text_length': len(text),
                    'estimated_tokens': text_tokens,
                    'used_chunking': needs_chunking,
                    'entities_found': len(detected_entities)
                }
            }
            
            results.append(result_doc)
        
        # Save results
        output_path = '/home/ide/ide/ECHR_mask_ai4privacy/output_ai4privacy.json'
        print(f"Saving {len(results)} results to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("✅ AI4Privacy processing complete!")
        print(f"📊 Statistics:")
        print(f"  - Total documents: {len(results)}")
        print(f"  - Documents requiring chunking: {chunked_docs}")
        print(f"  - Total entities found: {sum(len(doc['ai4privacy_detected_pii']) for doc in results)}")
        print(f"Results saved to: {output_path}")
        print("Run ai4privacy_evaluate.py to analyze performance")
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()