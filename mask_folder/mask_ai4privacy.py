
import pandas as pd
from ai4privacy import protect
import torch
import unicodedata
import re
torch.cuda.empty_cache()
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


input_file = "input_10.xlsx"
output_file = "output_masked_ai4privacy.xlsx"

df = pd.read_excel(input_file)
df["original_text"] = df["original_text"].astype(str)

all_masked_texts = []
all_detected_entities = []


for idx, text in enumerate(df["original_text"], 1):
    print(f"Processing row {idx}/{len(df)}...")

    norm_text = normalize_text(text)

    # Always use verbose=True for more info
    result = protect(
        norm_text,
        classify_pii=True,
        verbose=True  # changed to True
    )

    # Debug: print type and content if needed
    # print(type(result), result)

    if isinstance(result, dict):
        masked_text = result.get("text", "")
        entities = result.get("replacements", [])
    else:
        masked_text = result
        entities = []

    all_masked_texts.append(masked_text)
    all_detected_entities.append(entities)

    print(f"  Masked length: {len(masked_text)}, Detected PII types: {[e['label'] for e in entities]}")

# Save to Excel
df["masked_text"] = all_masked_texts
df["detected_pii"] = all_detected_entities
df.to_excel(output_file, index=False)
print(f"\nMasking completed. Results saved to {output_file}")
