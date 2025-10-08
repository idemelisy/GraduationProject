import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import unicodedata
import re

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Convert to standard unicode
    text = unicodedata.normalize("NFC", text)

    # 2. Replace Windows line endings (\r\n) with \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Replace multiple spaces/newlines with single ones
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # 4. Strip leading/trailing spaces
    text = text.strip()
    return text

# === Load Piiranha ===
model_name = "iiiorg/piiranha-v1-detect-personal-information"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipe = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  # merge subwords into full entity
)

# === Chunker ===
def chunk_text(text, max_tokens=400, stride=50):
    # Use the AI4Privacy tokenizer for token IDs
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded["input_ids"]

    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)

        if end == len(input_ids):
            break
        start += max_tokens - stride
    return chunks


# === Excel input/output ===
input_file = "input_10.xlsx"   # your Excel file
output_file = "output_piranha.xlsx" # result file

df = pd.read_excel(input_file)
df["original_text"] = df["original_text"].apply(normalize_text)
detected_list = []
masked_list = []

for text in df["original_text"]:
    all_entities = []
    masked_chunks = []
    chunks = chunk_text(text, max_tokens=400, stride=50)
    
    for chunk in chunks:
        preds = ner_pipe(chunk)
        
        # Create masked version of chunk
        masked_chunk = chunk
        spans_to_mask = []
        
        for p in preds:
            all_entities.append(p)
            # Collect spans for masking (reverse order to avoid offset issues)
            spans_to_mask.append((p['start'], p['end'], p['entity_group']))
        
        # Apply masking in reverse order
        for start, end, label in sorted(spans_to_mask, key=lambda x: x[0], reverse=True):
            masked_chunk = masked_chunk[:start] + f"[{label}]" + masked_chunk[end:]
        
        masked_chunks.append(masked_chunk)

    # Store results
    detected_list.append(str(all_entities))
    masked_list.append(" ".join(masked_chunks))

df["Detected_PII"] = detected_list
df["Piranha_Masked"] = masked_list
df.to_excel(output_file, index=False)
print(f"Detection finished. Results saved to {output_file}")
