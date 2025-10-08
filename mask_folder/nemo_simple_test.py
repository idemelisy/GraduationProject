import pandas as pd
import torch
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
import unicodedata
import re

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_entities_accurate(original_text, masked_text):
    """
    Extract PII entities by doing precise character-by-character alignment
    """
    import re
    detected_entities = []
    
    # Entity patterns and their lengths
    entity_patterns = {
        '<PERSON>': 'PERSON',
        '<DATE_TIME>': 'DATE_TIME',
        '<ADDRESS>': 'ADDRESS'
    }
    
    # Simple approach: split both texts into words and align them
    orig_words = re.findall(r'\S+', original_text)
    mask_words = re.findall(r'\S+', masked_text)
    
    # Track character position in original text
    orig_char_pos = 0
    orig_word_idx = 0
    mask_word_idx = 0
    
    while orig_word_idx < len(orig_words) and mask_word_idx < len(mask_words):
        orig_word = orig_words[orig_word_idx]
        mask_word = mask_words[mask_word_idx]
        
        # Find current word's position in original text
        word_start = original_text.find(orig_word, orig_char_pos)
        if word_start == -1:
            orig_word_idx += 1
            continue
        
        word_end = word_start + len(orig_word)
        
        # Check if masked word is an entity pattern
        if mask_word in entity_patterns:
            entity_type = entity_patterns[mask_word]
            
            # Extract the original entity text
            entity_text = orig_word
            
            # For multi-word entities, check if next words also belong to this entity
            # This is tricky without more context, so we'll keep it simple for now
            
            detected_entities.append({
                "label": entity_type,
                "start": word_start,
                "end": word_end,
                "text": entity_text
            })
            
        elif orig_word == mask_word:
            # Words match, continue
            pass
        else:
            # Words don't match - this might be part of a multi-word entity
            # Skip for now - this is the complex case
            pass
        
        orig_char_pos = word_end
        orig_word_idx += 1
        mask_word_idx += 1
    
    # Remove duplicates
    unique_entities = []
    seen = set()
    for entity in detected_entities:
        key = (entity['label'], entity['start'], entity['end'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    
    return unique_entities

def main():
    # ----------------------
    # Start Dask cluster
    # ----------------------
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print(client)

    # ----------------------
    # Excel input/output
    # ----------------------
    input_file = "input_10.xlsx"
    output_file = "output_nemo_simple.xlsx"

    df = pd.read_excel(input_file)
    df["original_text"] = df["original_text"].apply(normalize_text)

    all_masked_texts = []
    all_detected_entities = []

    # ----------------------
    # Iterate over rows - just first few for testing
    # ----------------------
    for idx, text in enumerate(df["original_text"][:3]):  # Test with just 3 rows
        print(f"\nProcessing row {idx+1}...")
        print(f"Original text (first 100 chars): {text[:100]}...")
        
        df_chunk = pd.DataFrame({'text': [text]})
        ddf_chunk = dd.from_pandas(df_chunk, npartitions=1)
        dataset = DocumentDataset(ddf_chunk)

        # Create modifier
        modifier = PiiModifier(
            language="en",
            supported_entities=["PERSON", "ADDRESS", "DATE_TIME"],
            anonymize_action="replace",
            return_decision=True,
            batch_size=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        modify_pipeline = Modify(modifier)
        masked_dataset = modify_pipeline(dataset)
        masked_df = masked_dataset.df.compute()

        # Get masked text
        masked_text = masked_df['text'].iloc[0]
        all_masked_texts.append(masked_text)
        
        print(f"Masked text (first 100 chars): {masked_text[:100]}...")

        # Count entity placeholders in masked text
        person_count = masked_text.count('<PERSON>')
        date_count = masked_text.count('<DATE_TIME>')
        address_count = masked_text.count('<ADDRESS>')
        print(f"Entity counts - PERSON: {person_count}, DATE_TIME: {date_count}, ADDRESS: {address_count}")

        # Extract entities using simple method
        entities = extract_entities_accurate(text, masked_text)
        all_detected_entities.append(entities)
        
        print(f"Extracted {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity['label']}: '{entity['text']}' at {entity['start']}-{entity['end']}")

    # Close cluster
    client.close()
    cluster.close()

    # ----------------------
    # Create simple output for testing
    # ----------------------
    test_df = df.head(3).copy()
    test_df["Masked_Text"] = all_masked_texts
    test_df["Detected_Entities"] = all_detected_entities
    
    # Create readable format
    formatted_entities = []
    for entities in all_detected_entities:
        if entities:
            formatted_list = []
            for entity in entities:
                formatted_list.append(f"Label: {entity['label']}, Start: {entity['start']}, End: {entity['end']}, Text: '{entity['text']}'")
            formatted_entities.append(" | ".join(formatted_list))
        else:
            formatted_entities.append("No PII detected")
    
    test_df["Formatted_Entities"] = formatted_entities
    test_df.to_excel(output_file, index=False)
    
    print(f"\nTest completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()