#!/usr/bin/env python3
"""
nemo_script.py

Runs NeMo Curator PII Modifier on a dataset and outputs detected PII spans in JSON format.
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
# HELPER: Extract entities from masked vs original
# -----------------------------------------------------
def extract_pii_entities(original, masked):
    entities = []
    pattern = re.compile(r"(\{\{(.*?)\}\}|<(.*?)>)")
    orig_idx = 0
    masked_idx = 0
    while masked_idx < len(masked):
        match = pattern.search(masked, masked_idx)
        if not match:
            break
        entity_type = match.group(2) if match.group(2) else match.group(3)
        start_masked = match.start()
        pre_masked = masked[masked_idx:start_masked]
        orig_idx = original.find(pre_masked, orig_idx)
        if orig_idx == -1:
            break
        orig_idx += len(pre_masked)
        next_masked = match.end()
        next_entity_start = pattern.search(masked, next_masked)
        if next_entity_start:
            post_masked = masked[match.end():next_entity_start.start()]
        else:
            post_masked = masked[match.end():]
        end_idx = original.find(post_masked, orig_idx) if post_masked else len(original)
        entity_text = original[orig_idx:end_idx].strip()
        entities.append({
            "label": entity_type,
            "text": entity_text,
            "start": orig_idx,
            "end": end_idx
        })
        masked_idx = match.end()
        orig_idx = end_idx
    return entities

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    input_path = "/home/ide/ide/ECHR_mask/annotations.json"   # your input dataset
    output_path = "/home/ide/ide/ECHR_mask_nemo/output_nemo.json"

    # Load your dataset (expected format: [{"id":..., "text":...}, ...])
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Start Dask
    client = Client()
    print("Dask client started:", client)

    results = []

    # Configure NeMo modifier
    modifier = PiiModifier(
        language="en",
        supported_entities=["PERSON", "ADDRESS", "DATE_TIME", "EMAIL_ADDRESS", "PHONE_NUMBER", "ORGANIZATION"],
        anonymize_action="replace",
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    modify_pipeline = Modify(modifier)

    # Process one by one (Dask used internally by Modify)
    for i, item in enumerate(data):
        text = item["text"]
        df = pd.DataFrame({"text": [text]})
        ddf = dd.from_pandas(df, npartitions=1)
        dataset = DocumentDataset(ddf)

        masked_dataset = modify_pipeline(dataset)
        masked_text = masked_dataset.df.compute()["text"].iloc[0]

        detected_entities = extract_pii_entities(text, masked_text)
        results.append({
            "id": item.get("id", f"doc_{i}"),
            "text": text,
            "nemo_masked_text": masked_text,
            "nemo_detected_pii": detected_entities
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(data)} documents...")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Saved detections to {output_path}")
