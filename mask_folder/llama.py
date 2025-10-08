import pandas as pd
from transformers import pipeline
import unicodedata, re, json

# --- normalization ---
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# --- chunking ---
def chunk_text(text, max_tokens=400, stride=50):
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        chunks.append(tokenizer.decode(input_ids[start:end], clean_up_tokenization_spaces=True))
        if end == len(input_ids): break
        start += max_tokens - stride
    return chunks

# --- LLaMA pipeline ---
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B", device=0)
valid_PII_classes = ['GIVENNAME', 'SURNAME', 'STREET', 'DATEOFBIRTH', 'CITY']

def detect_pii_with_llama(chunk):
    prompt = f"""
    Extract all personally identifiable information (PII) from the text below.
    Return a JSON dictionary with keys {valid_PII_classes}.
    Text: \"\"\"{chunk}\"\"\"
    """
    output = pipe(prompt, max_new_tokens=256)[0]['generated_text']
    import re
    match = re.search(r'\{.*\}', output, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except: return {k: [] for k in valid_PII_classes}
    return {k: [] for k in valid_PII_classes}

# --- Excel processing ---
input_file = "input_10.xlsx"
output_file = "output_llama.xlsx"
df = pd.read_excel(input_file)
df["original_text"] = df["original_text"].apply(normalize_text)

detected_list = []
masked_list = []

for text in df["original_text"]:
    all_entities = []
    masked_chunks = []
    chunks = chunk_text(text, max_tokens=400, stride=50)
    
    for chunk in chunks:
        pii_dict = detect_pii_with_llama(chunk)
        all_entities.append(pii_dict)
        masked_chunk = chunk
        for pii_type, values in pii_dict.items():
            for val in values:
                masked_chunk = masked_chunk.replace(val, f'[{pii_type}]')
        masked_chunks.append(masked_chunk)
    
    detected_list.append(all_entities)
    masked_list.append(" ".join(masked_chunks))

df["Detected_PII"] = detected_list
df["LLaMA_Masked"] = masked_list
df.to_excel(output_file, index=False)
print(f"Detection finished. Results saved to {output_file}")
