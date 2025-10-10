#!/usr/bin/env python3
"""
ai4privacy_simplified.py
Simpler version of the AI4Privacy chunked processor for ECHR documents.
"""

import json, re
from typing import List, Dict
from ai4privacy import protect
from transformers import AutoTokenizer

class AI4PrivacyProcessor:
    def __init__(self, model_name="ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"):
        self.max_tokens = 1536
        self.chunk_size = 1200
        self.overlap = 150
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def sentence_split(self, text: str) -> List[str]:
        return re.split(r'(?<=[.!?])\s+', text.strip())

    def make_chunks(self, text: str) -> List[Dict]:
        sentences = self.sentence_split(text)
        chunks, current, token_count = [], [], 0

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)
            if token_count + sent_tokens > self.chunk_size and current:
                chunks.append(" ".join(current))
                # overlap last few sentences
                current = current[-3:]  # simple fixed overlap
                token_count = self.count_tokens(" ".join(current))
            current.append(sent)
            token_count += sent_tokens
        if current:
            chunks.append(" ".join(current))
        return chunks

    def run_chunk(self, text: str, offset: int = 0) -> List[Dict]:
        out = protect(text, classify_pii=True, verbose=True)

        ents = []
        
        # Debug: Check what protect() actually returns
        print(f"Type of out: {type(out)}")
        print(f"Content of out: {out}")
        
        # Handle case where protect returns a string instead of dict
        if isinstance(out, str):
            # If it's just the anonymized text, we can't extract entities
            print(f"Warning: protect() returned string instead of dict: {out[:100]}...")
            return []
        
        # Handle dictionary response
        if isinstance(out, dict):
            for r in out.get("replacements", []):
                ents.append({
                    "label": r["label"],
                    "text": r["value"],
                    "start": r["start"] + offset,
                    "end": r["end"] + offset
                })
        
        return ents

    def detect_pii(self, text: str) -> List[Dict]:
        if self.count_tokens(text) <= self.chunk_size:
            return self.run_chunk(text)
        entities, offset = [], 0
        for chunk in self.make_chunks(text):
            entities += self.run_chunk(chunk, offset)
            offset += len(chunk)  # approximate alignment
        # remove duplicates by position/label
        seen, clean = set(), []
        for e in sorted(entities, key=lambda x: (x["start"], x["end"])):
            key = (e["label"], e["start"], e["end"])
            if key not in seen:
                seen.add(key)
                clean.append(e)
        return clean

def main():
    print("=== AI4Privacy Simplified Processing ===")
    with open("/home/ide/ide/data/input.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    proc = AI4PrivacyProcessor()
    results = []

    for i, doc in enumerate(data):
        text = doc["text"]
        doc_id = doc.get("doc_id", str(i))
        entities = proc.detect_pii(text)
        results.append({
            "id": doc_id,
            "text": text,
            "ai4privacy_detected_pii": entities,
            "annotations": doc.get("annotations", [])
        })
        if i % 50 == 0:
            print(f"Processed {i+1}/{len(data)}")

    out = "/home/ide/ide/ECHR_mask_ai4privacy/output_ai4privacy.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Done. Saved results to {out}")

if __name__ == "__main__":
    main()
