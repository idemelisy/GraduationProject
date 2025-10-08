import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import unicodedata
import re

# ----------------------
# 1️⃣ Normalization function
# ----------------------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# ----------------------
# 2️⃣ Chunking function
# ----------------------
def chunk_text(text, max_tokens=400, stride=50, tokenizer=None):
    if tokenizer is None:
        # naive word-based chunking
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += max_tokens - stride
        return chunks
    else:
        # tokenizer-based chunking (optional)
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        chunks = []
        start = 0
        while start < len(input_ids):
            end = min(start + max_tokens, len(input_ids))
            chunk_ids = input_ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids)
            chunks.append(chunk_text)
            if end == len(input_ids):
                break
            start += max_tokens - stride
        return chunks

# ----------------------
# 3️⃣ Load Excel
# ----------------------
input_file = "input_10.xlsx"
output_file = "output_hf.xlsx"

df = pd.read_excel(input_file)

# Apply normalization
df["original_text"] = df["original_text"].apply(normalize_text)

# ----------------------
# 4️⃣ Load HuggingFace NER pipeline
# ----------------------
# You can replace "dbmdz/bert-large-cased-finetuned-conll03-english" with any NER model
ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

ner_pipe = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  # merge subwords
)

# ----------------------
# 5️⃣ Masking function for a chunk
# ----------------------
def mask_chunk(text):
    if not text.strip():
        return text, []

    entities = ner_pipe(text)
    spans = []
    for ent in entities:
        if 'start' in ent and 'end' in ent and 'entity_group' in ent:
            spans.append((ent['start'], ent['end'], ent['entity_group']))

    # Mask text
    masked = list(text)
    for start, end, label in sorted(spans, reverse=True):
        masked[start:end] = f"[{label}]"

    # Detected entities
    detected = [{"text": ent['word'], "start": ent['start'], "end": ent['end'], "label": ent['entity_group']} 
                for ent in entities]

    return ''.join(masked), detected

# ----------------------
# 6️⃣ Apply chunking + masking
# ----------------------
all_masked_texts = []
all_detected_pii = []

for text in df["original_text"]:
    # Use tokenizer-based chunking for better performance with BERT models
    chunks = chunk_text(text, max_tokens=400, stride=50, tokenizer=tokenizer)
    masked_chunks = []
    detected_chunks = []

    for chunk in chunks:
        masked_chunk, detected_chunk = mask_chunk(chunk)
        masked_chunks.append(masked_chunk)
        detected_chunks.extend(detected_chunk)

    # Merge chunks back to full text
    full_masked_text = " ".join(masked_chunks)
    all_masked_texts.append(full_masked_text)
    all_detected_pii.append(detected_chunks)

# ----------------------
# 7️⃣ Save results
# ----------------------
df["hf_masked"] = all_masked_texts
df["hf_detected_pii"] = all_detected_pii
df.to_excel(output_file, index=False)
print(f"Done. Masked text and detected PII saved to {output_file}")
