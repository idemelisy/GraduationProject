#!/usr/bin/env python3
"""
evaluate_ai4privacy.py

Evaluate AI4Privacy output JSON against ECHR ground truth annotations.
Compatible with Piranha-style evaluation output and logic.
"""

import json
import argparse
from typing import List

# AI4Privacy → GT mappings
PII_TO_GT = {
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON", 
    "TITLE": "PERSON",
    "CITY": "LOC",
    "STREET": "LOC",
    "ZIPCODE": "LOC",
    "BUILDINGNUM": "LOC",
    "DATE": "DATETIME",
    "TIME": "DATETIME",
}

# Flexible label matching - consider these as equivalent for evaluation
EQUIVALENT_LABELS = {
    ("LOC", "ORG"): True,  # Location and Organization can be equivalent
    ("ORG", "LOC"): True,  # (e.g., "Copenhagen City Court" could be either)
}

# Ignore sets
IGNORE_FP_LABELS = {
    "EMAIL", "TAXNUM", "TELEPHONENUM", "SOCIALNUM",
    "BANKACCOUNTNUM", "IBAN", "DRIVERLICENSENUM",
    "SEX", "AGE", "GENDER", "IDCARDNUM"
}
IGNORE_GT_LABELS = {"CODE", "CASE", "COURT", "QUANTITY", "MISC"}
IGNORE_FN_LABELS = {"ORG", "DEM", "QUANTITY", "MISC", "CODE", "CASE", "COURT"}


# --- Utility functions ---

def iou_span(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0, inter_end - inter_start)
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def merge_consecutive_entities(pred_list):
    """Merge consecutive predictions of same type separated by tiny gaps."""
    if not pred_list:
        return []
    pred_list.sort(key=lambda x: x["start"])
    merged = [pred_list[0].copy()]
    for next_pred in pred_list[1:]:
        current = merged[-1]
        gap = next_pred["start"] - current["end"]
        
        # For DATE/TIME/DATETIME entities, allow larger gaps (up to 3 chars for spaces/punctuation)
        # For other entities, use the original gap <= 1 logic
        datetime_labels = ["DATE", "TIME", "DATETIME"]
        max_gap = 3 if current["label"] in datetime_labels and next_pred["label"] in datetime_labels else 1
        
        if current["label"] == next_pred["label"] and gap <= max_gap:
            current["end"] = next_pred["end"]
            current["pred_text"] += " " + next_pred["pred_text"]
        elif (current["label"] in datetime_labels and next_pred["label"] in datetime_labels and gap <= 3):
            # Merge different date/time components (e.g., DATE + TIME, or day + month + year)
            current["end"] = next_pred["end"]
            current["pred_text"] += " " + next_pred["pred_text"]
            # Keep the current label (don't change DATETIME back to DATE)
        else:
            merged.append(next_pred.copy())
    return merged


def merge_name_entities(pred_list: List[dict]) -> List[dict]:
    """Merge consecutive GIVENNAME + SURNAME into a single PERSON."""
    merged = []
    current = None
    for p in pred_list:
        label = PII_TO_GT.get(p["label"], p["label"])
        if label == "PERSON":
            if current is None:
                current = p.copy()
                current["label"] = label
            elif current["end"] >= p["start"] - 1:
                current["end"] = p["end"]
                current["pred_text"] += " " + p["pred_text"]
            else:
                merged.append(current)
                current = p.copy()
                current["label"] = label
        else:
            if current is not None:
                merged.append(current)
                current = None
            merged.append(p)
    if current is not None:
        merged.append(current)
    return merged


def labels_match(gt_label, pred_label):
    """Check if ground truth and predicted labels should be considered a match."""
    if gt_label == pred_label:
        return True
    return EQUIVALENT_LABELS.get((gt_label, pred_label), False)


def is_partial_datetime_match(gt_start, gt_end, pred_start, pred_end, gt_label, pred_label):
    """
    Check if a predicted DATETIME is a partial match within a larger ground truth DATETIME.
    This handles cases where AI4Privacy detects individual dates within a date range.
    Allow small overflows (1-2 chars) for punctuation differences.
    """
    if gt_label != "DATETIME" or pred_label != "DATETIME":
        return False
    
    # Check if predicted span is mostly contained within ground truth span
    # Allow small overflow for punctuation (e.g., "1995," vs "1995")
    overlap_start = max(gt_start, pred_start)
    overlap_end = min(gt_end, pred_end)
    overlap = max(0, overlap_end - overlap_start)
    pred_length = pred_end - pred_start
    
    # Consider it a partial match if:
    # 1. Most of the prediction overlaps with GT (at least 80%)
    # 2. The prediction starts within or slightly before GT span
    # 3. The prediction doesn't extend too far beyond GT span (max 2 chars for punctuation)
    overlap_ratio = overlap / pred_length if pred_length > 0 else 0
    
    return (overlap_ratio >= 0.8 and 
            pred_start >= gt_start - 2 and 
            pred_end <= gt_end + 2)


def evaluate_document(doc, iou_threshold=0.5):
    text = doc["text"]
    all_annotations = doc.get("annotations", [])

    # --- Use single annotator if multiple ---
    annotators = sorted(set(a.get("annotator") for a in all_annotations if "annotator" in a))
    if annotators:
        chosen_annotator = annotators[0]
        gt_raw = [a for a in all_annotations if a.get("annotator") == chosen_annotator]
    else:
        gt_raw = all_annotations

    # Fix misaligned annotations by finding correct positions in text
    gt_list = []
    seen_positions = set()  # Track seen positions to avoid duplicates
    
    for a in gt_raw:
        if a["entity_type"] in IGNORE_GT_LABELS:
            continue
        
        # Get expected text and annotated offsets
        expected_text = a.get("span_text", "")
        annotated_start = a["start_offset"]
        annotated_end = a["end_offset"]
        actual_text_at_offset = text[annotated_start:annotated_end]
        
        # Check if the offset is correct
        if expected_text and actual_text_at_offset.strip() == expected_text.strip():
            # Offset is correct, use as-is
            start_pos = annotated_start
            end_pos = annotated_end
            gt_text = expected_text
        elif expected_text:
            # Offset is wrong, try to find the correct position
            # Look for all occurrences and pick the closest to the annotated position
            search_start = 0
            best_pos = -1
            min_distance = float('inf')
            
            while True:
                pos = text.find(expected_text, search_start)
                if pos == -1:
                    break
                distance = abs(pos - annotated_start)
                if distance < min_distance:
                    min_distance = distance
                    best_pos = pos
                search_start = pos + 1
            
            if best_pos != -1:
                start_pos = best_pos
                end_pos = best_pos + len(expected_text)
                gt_text = expected_text
            else:
                # Try with stripped text
                stripped_expected = expected_text.strip()
                pos = text.find(stripped_expected)
                if pos != -1:
                    start_pos = pos
                    end_pos = pos + len(stripped_expected)
                    gt_text = stripped_expected
                else:
                    # Can't find the text, skip this annotation
                    print(f"Warning: Could not find '{expected_text}' in text, skipping annotation")
                    continue
        else:
            # No span_text available, use offset-based extraction
            start_pos = annotated_start
            end_pos = annotated_end
            gt_text = actual_text_at_offset
        
        # Check for duplicates at the same position with same label
        position_key = (start_pos, end_pos, a["entity_type"])
        if position_key in seen_positions:
            continue  # Skip duplicate
        seen_positions.add(position_key)
            
        gt_list.append({
            "start": start_pos,
            "end": end_pos,
            "label": a["entity_type"],
            "gt_text": gt_text,
        })

    pred_list = merge_name_entities([
        {
            "start": p["start"],
            "end": p["end"],
            "label": PII_TO_GT.get(p["label"], p["label"]),
            "pred_text": p["text"],
            "used": False
        }
        for p in doc.get("ai4privacy_detected_pii", [])
        if p["label"] not in IGNORE_FP_LABELS
    ])
    pred_list = merge_consecutive_entities(pred_list)

    tp, fn, fp = [], [], []
    gt_matched = [False] * len(gt_list)

    # Track which predictions are used to avoid double-counting
    for p in pred_list:
        p["used"] = False

    # For partial datetime matching, we allow multiple predictions to match the same GT
    # Keep track of which GTs have been matched by partial datetime matches
    gt_partial_matched = [False] * len(gt_list)

    for pi, p in enumerate(pred_list):
        if p["used"]:
            continue
            
        for gi, gt in enumerate(gt_list):
            inter = iou_span(gt["start"], gt["end"], p["start"], p["end"])
            
            # Standard IOU-based matching (exclusive - only one prediction per GT)
            standard_match = (inter >= iou_threshold and 
                            labels_match(gt["label"], p["label"]) and 
                            not gt_matched[gi])
            
            # Partial datetime matching (multiple predictions can match same GT)
            partial_datetime_match = (inter > 0 and 
                                    is_partial_datetime_match(gt["start"], gt["end"], 
                                                            p["start"], p["end"], 
                                                            gt["label"], p["label"]))
            
            if standard_match:
                gt_matched[gi] = True
                p["used"] = True
                tp.append({
                    "start": gt["start"],
                    "end": gt["end"],
                    "label": gt["label"],
                    "ground_truth_text": gt["gt_text"],
                    "predicted_text": p["pred_text"]
                })
                break
                
            elif partial_datetime_match and not gt_matched[gi]:
                # For partial matches, don't mark GT as matched (allow multiple predictions)
                # But do mark this GT as having partial matches
                gt_partial_matched[gi] = True
                p["used"] = True
                tp.append({
                    "start": gt["start"],
                    "end": gt["end"],
                    "label": gt["label"],
                    "ground_truth_text": gt["gt_text"],
                    "predicted_text": p["pred_text"]
                })
                break

    # Now mark GTs that had partial matches as matched
    for gi in range(len(gt_list)):
        if gt_partial_matched[gi]:
            gt_matched[gi] = True

    # False negatives (unmatched GT)
    for gi, gt in enumerate(gt_list):
        if not gt_matched[gi] and gt["label"] not in IGNORE_FN_LABELS:
            fn.append({
                "start": gt["start"],
                "end": gt["end"],
                "label": gt["label"],
                "ground_truth_text": gt["gt_text"]
            })

    # False positives (unused preds)
    for p in pred_list:
        if not p["used"]:
            fp.append({
                "start": p["start"],
                "end": p["end"],
                "label": p["label"],
                "predicted_text": p["pred_text"]
            })

    return {"true_positives": tp, "false_negatives": fn, "false_positives": fp}


def aggregate_results(all_docs):
    tp = sum(len(d["true_positives"]) for d in all_docs)
    fn = sum(len(d["false_negatives"]) for d in all_docs)
    fp = sum(len(d["false_positives"]) for d in all_docs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main(args):
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = data["documents"] if isinstance(data, dict) and "documents" in data else data

    detailed = []
    for doc in documents:
        eval_doc = evaluate_document(doc, iou_threshold=args.iou_threshold)
        detailed.append({"doc_id": doc.get("doc_id"), "evaluation": eval_doc})

    agg = aggregate_results([d["evaluation"] for d in detailed])

    print("===== AI4Privacy Evaluation (Legal Text Adapted) =====")
    print(f"Documents evaluated : {len(detailed)}")
    print(f"TP : {agg['true_positives']}")
    print(f"FP : {agg['false_positives']}")
    print(f"FN : {agg['false_negatives']}")
    print(f"Precision : {agg['precision']:.4f}")
    print(f"Recall    : {agg['recall']:.4f}")
    print(f"F1        : {agg['f1']:.4f}")

    if args.out_detailed:
        with open(args.out_detailed, "w", encoding="utf-8") as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)
        print(f"Detailed per-document evaluation saved to: {args.out_detailed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI4Privacy output")
    parser.add_argument("--input", type=str, default="output_ai4privacy.json")
    parser.add_argument("--out_detailed", type=str, default="detailed_ai4privacy_eval.json")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
