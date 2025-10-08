import pandas as pd
import unicodedata
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

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
def chunk_text(text, max_tokens=400, stride=50):
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

# ----------------------
# 3️⃣ Load Excel
# ----------------------
input_file = "input_10.xlsx"
output_file = "output_presidio.xlsx"

df = pd.read_excel(input_file)

# Apply normalization
df["original_text"] = df["original_text"].apply(normalize_text)

# ----------------------
# 4️⃣ Initialize Presidio engines
# ----------------------
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# ----------------------
# 5️⃣ Masking function with chunking
# ----------------------
def presidio_mask_and_extract(text):
    if not text.strip():
        return "", []

    chunks = chunk_text(text, max_tokens=400, stride=50)
    masked_chunks = []
    detected_entities = []

    for chunk in chunks:
        # Analyze
        results = analyzer.analyze(text=chunk, language="en")

        # Collect detected PII
        for r in results:
            detected_entities.append({
                "label": r.entity_type,
                "text": chunk[r.start:r.end]
            })

        # Prepare operators for anonymization
        operators = {r.entity_type: OperatorConfig("replace", {"new_value": f"[{r.entity_type}]"})
                     for r in results}

        if not operators:
            operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "[PII]"})}

        # Anonymize / mask
        masked_chunk = anonymizer.anonymize(
            text=chunk,
            analyzer_results=results,
            operators=operators
        ).text

        masked_chunks.append(masked_chunk)

    # Merge chunks back to full text
    full_masked_text = " ".join(masked_chunks)
    return full_masked_text, detected_entities

# ----------------------
# 6️⃣ Apply masking to all rows
# ----------------------
all_masked = []
all_detected = []

for text in df["original_text"]:
    masked_text, entities = presidio_mask_and_extract(text)
    all_masked.append(masked_text)
    all_detected.append(entities)

df["presidio_masked"] = all_masked
df["presidio_detected_pii"] = all_detected

# ----------------------
# 7️⃣ Save to Excel
# ----------------------
df.to_excel(output_file, index=False)
print(f"Presidio masking and PII extraction completed. Output saved to {output_file}")
