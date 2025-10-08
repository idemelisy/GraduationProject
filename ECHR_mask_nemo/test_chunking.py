#!/usr/bin/env python3
"""
test_chunking.py - Debug the chunking function
"""

def chunk_text(text, max_chunk_size=2000, overlap=200):
    """
    Split text into overlapping chunks to avoid processing issues.
    """
    print(f"Chunking text of length {len(text)}")
    
    if len(text) <= max_chunk_size:
        result = [(text, 0)]
        print(f"Text fits in one chunk: {result}")
        return result
    
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
        print(f"Created chunk {len(chunks)}: start={start}, end={end}, length={len(chunk)}")
        
        if end >= len(text):
            break
            
        # Move start forward, but with overlap
        start = end - overlap
    
    return chunks

# Test with a sample text
test_text = "This is a test sentence. Here is another one! And a third? Now we continue with more text to test the chunking functionality. This should create multiple chunks when the text is long enough."

chunks = chunk_text(test_text, max_chunk_size=50, overlap=10)
print(f"\nFinal result: {len(chunks)} chunks")
for i, (chunk, offset) in enumerate(chunks):
    print(f"Chunk {i+1}: offset={offset}, text='{chunk[:50]}...'")