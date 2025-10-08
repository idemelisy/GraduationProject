import json
from datasets import load_dataset
from huggingface_hub import login
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ---- 1. Login to Hugging Face Hub ----
# SAFER: run `huggingface-cli login` once in your terminal instead of hardcoding your token
# login("hf_xxx")   # uncomment if you want to hardcode (not recommended)

# ---- 2. Load dataset from HuggingFace ----
dataset = load_dataset("mattmdjaga/text-anonymization-benchmark-train")
print("Sample entry:", dataset["train"][0])

# ---- 3. Initialize Presidio ----
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
ENTITY_MAPPING = {
    "DATE_TIME": "DATETIME",
    "NRP": "DEM",
    "PERSON": "PERSON",
    "LOC": "LOC"
    # extend if needed
}

def normalize_entity(label: str) -> str:
    """Map Presidio entity to ground truth schema."""
    return ENTITY_MAPPING.get(label, label)

def evaluate_detection(doc):
    """Compare Presidio detections with ground truth annotations using mapping + text spans."""
    gt = doc.get("annotations", [])
    pred = doc.get("presidio_detected_pii", [])
    
    matches, false_negatives, false_positives = [], [], []
    text = doc["text"]

    # Normalize labels before comparison
    gt_spans = {
        (a["start_offset"], a["end_offset"], a["entity_type"]): a for a in gt
    }
    pred_spans = {
        (p["start"], p["end"], normalize_entity(p["label"])): p for p in pred
    }
    
    # True positives
    for span, a in gt_spans.items():
        if span in pred_spans:
            p = pred_spans[span]
            matches.append({
                "start": span[0],
                "end": span[1],
                "label": span[2],
                "ground_truth_text": text[span[0]:span[1]],
                "presidio_spanned_text": p["text"]
            })
        else:
            false_negatives.append({
                "start": span[0],
                "end": span[1],
                "label": span[2],
                "ground_truth_text": text[span[0]:span[1]]
            })

    # False positives
    for span, p in pred_spans.items():
        if span not in gt_spans:
            false_positives.append({
                "start": span[0],
                "end": span[1],
                "label": span[2],
                "presidio_spanned_text": text[span[0]:span[1]]
            })

    return {
        "true_positives": matches,
        "false_negatives": false_negatives,
        "false_positives": false_positives
    }


def presidio_anon(text: str):
    """Analyze and anonymize text with Presidio."""
    if not isinstance(text, str) or not text.strip():
        return text, []
    
    # Step 1: Analyze text for PII
    results = analyzer.analyze(text=text, language="en")
    
    # Step 2: Operator configs: replace each entity with [ENTITY_TYPE]
    operators = {
        r.entity_type: OperatorConfig("replace", {"new_value": f"[{r.entity_type}]"})
        for r in results
    }
    if not operators:
        operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "[PII]"})}
    
    # Step 3: Anonymize
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )
    
    return anonymized.text, results


# ---- 4. Load JSON file (annotations.json) ----
with open("annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- 5. Apply Presidio ----
for doc in data:
    masked, detected = presidio_anon(doc["text"])
    doc["presidio_anonymized_text"] = masked
    doc["presidio_detected_pii"] = [
        {
            "label": r.entity_type,
            "text": doc["text"][r.start:r.end],
            "start": r.start,
            "end": r.end
        }
        for r in detected
    ]
    doc["evaluation"] = evaluate_detection(doc)
# ---- 6. Save output ----
with open("output_presidio_eval.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Done! Anonymized JSON saved to output_presidio_eval.json")
