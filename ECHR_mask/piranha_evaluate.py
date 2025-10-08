#!/usr/bin/env python3
"""
evaluate_piranha.py

Evaluate Piranha output JSON against ground truth annotations.
Handles merging of GIVENNAME + SURNAME → PERSON, and location/date mappings.
"""

import json
import argparse
from typing import List

# Piranha -> GT mappings
PII_TO_GT = {
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "CITY": "LOC",
    "BUILDINGNUM": "LOC",
    "STREET": "LOC",
    "ZIPCODE": "LOC",
    "DATEOFBIRTH": "DATETIME"
}

# Labels to ignore FP/FN
IGNORE_FP_LABELS = set()  # you may add others if needed

# Ground truth labels to ignore completely (exclude from evaluation)
IGNORE_GT_LABELS = {"CODE", "CASE", "COURT", "QUANTITY", "MISC"}

# Additional labels to ignore only for false negatives (Piranha not expected to detect these)
IGNORE_FN_LABELS = {"ORG", "DEM", "QUANTITY", "MISC", "CODE", "CASE", "COURT", "DATETIME"}

def check_special_loc_org_match(pred_label: str, pred_start: int, pred_end: int, 
                               all_annotations: List, text: str) -> bool:
    """
    Check if a LOC prediction should match an ORG or DEM ground truth entity.
    This handles cases like:
    1. "Copenhagen" (LOC) in "Copenhagen City Court" (ORG)
    2. "Københavns" (LOC) in "Københavns Byret" (ORG) - court names with city names
    3. "Katowice" (LOC) in "Katowice District Prosecutor" (DEM) - institutions with city names
    """
    if pred_label not in ["LOC", "CITY"]:  # Handle both LOC and CITY labels
        return False
    
    # Check if this LOC prediction overlaps with an ORG or DEM annotation
    for ann in all_annotations:
        if ann["entity_type"] in ["ORG", "DEM"]:
            gt_start = ann["start_offset"]
            gt_end = ann["end_offset"]
            
            # Check if prediction overlaps with the ORG/DEM (not necessarily fully contained)
            if not (pred_end <= gt_start or pred_start >= gt_end):
                # Get the actual predicted text and clean it
                pred_text = text[pred_start:pred_end].strip()
                gt_text = ann["span_text"].strip()
                
                # Allow if predicted text is in the ground truth and is a meaningful location
                # This covers cases like "Copenhagen" in "Copenhagen City Court" 
                # AND "Københavns" in "Københavns Byret" AND "Katowice" in "Katowice District Prosecutor"
                if (pred_text.lower() in gt_text.lower() and 
                    len(pred_text) >= 3 and 
                    pred_text.replace(' ', '').replace('-', '').isalpha()):  # Allow spaces and hyphens but ensure it's alphabetic
                    
                    # Check for court/legal/institutional indicators
                    institution_indicators = ['court', 'byret', 'landsret', 'tribunal', 'højesteret', 'supreme', 
                                            'prosecutor', 'district', 'police', 'department', 'ministry', 'office']
                    has_indicator = any(indicator in gt_text.lower() for indicator in institution_indicators)
                    
                    if has_indicator:
                        return True
                    
                    # Also allow for general ORG/DEM entities containing location names
                    if len(pred_text) >= 4:  # Slightly stricter for non-institutional entities
                        return True
    
    return False
    
    return False

def iou_span(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0, inter_end - inter_start)
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0

def merge_consecutive_entities(pred_list):
    if not pred_list:
        return []
    pred_list.sort(key=lambda x: x["start"])
    merged = [pred_list[0].copy()]

    for next_pred in pred_list[1:]:
        current = merged[-1]
        gap = next_pred["start"] - current["end"]

        # Merge only if same label, small gap, and next text starts lowercase or punctuation
        if (current["label"] == next_pred["label"]
            and gap <= 1
            and (not next_pred["pred_text"] or next_pred["pred_text"][0].islower()
                 or next_pred["pred_text"][0] in " .,-")):
            current["end"] = next_pred["end"]
            current["pred_text"] += next_pred["pred_text"]
        else:
            merged.append(next_pred.copy())
    return merged


def merge_name_entities(pred_list: List[dict]) -> List[dict]:
    """Merge consecutive GIVENNAME + SURNAME tokens into one PERSON entity."""
    merged = []
    current = None
    for p in pred_list:
        label = PII_TO_GT.get(p["label"], p["label"])
        if label == "PERSON":
            if current is None:
                current = p.copy()
                current["label"] = label  # Update the label to PERSON
            elif current["end"] >= p["start"] - 1:  # Allow small gaps for consecutive names
                current["end"] = p["end"]
                current["pred_text"] += p["pred_text"]
            else:
                merged.append(current)
                current = p.copy()
                current["label"] = label  # Update the label to PERSON
        else:
            if current is not None:
                merged.append(current)
                current = None
            merged.append(p)
    if current is not None:
        merged.append(current)
    return merged

def evaluate_document(doc, iou_threshold=0.5):
    text = doc["text"]
    
    # --- Pick only one annotator's annotations per document ---
    all_annotations = doc.get("annotations", [])

    # Collect all annotator IDs
    annotators = sorted(set(a.get("annotator") for a in all_annotations if "annotator" in a))

    if annotators:
        chosen_annotator = annotators[0]  # pick the first one (or random.choice if you prefer)
        filtered_annotations = [a for a in all_annotations if a.get("annotator") == chosen_annotator]
    else:
        # No annotator info — fallback to all
        filtered_annotations = all_annotations

    gt_raw = [
        {
            "start": a["start_offset"],
            "end": a["end_offset"],
            "label": a["entity_type"],
            "gt_text": text[a["start_offset"]:a["end_offset"]]
        }
        for a in filtered_annotations
        if a["entity_type"] not in IGNORE_GT_LABELS
    ]
    
    # Remove duplicates by creating unique spans
    seen_spans = set()
    gt_list = []
    for gt in gt_raw:
        span_key = (gt["start"], gt["end"], gt["label"])
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            gt_list.append(gt)

    pred_list = merge_name_entities([
        {
            "start": p["start"],
            "end": p["end"],
            "label": PII_TO_GT.get(p["label"], p["label"]),
            "pred_text": p["text"],
            "used": False
        }
        for p in doc.get("piranha_detected_pii", [])
    ])
    
    # After name merging, also merge consecutive entities of the same type
    pred_list = merge_consecutive_entities(pred_list)

    tp, fn, fp = [], [], []
    # Track which ground truth entities are matched
    gt_matched = [False] * len(gt_list)

    for pi, p in enumerate(pred_list):
        matched_gts = []
        
        # First, try normal matching with ground truth
        for gi, gt in enumerate(gt_list):
            # Skip ground truth entities that are already matched
            if gt_matched[gi]:
                continue
                
            # Use different IoU thresholds for different entity types
            if gt["label"] == "LOC":
                threshold = 0.3
            elif gt["label"] == "PERSON":
                threshold = 0.4
            else:
                threshold = iou_threshold
                
            inter = iou_span(gt["start"], gt["end"], p["start"], p["end"])
            
            # For same-label matches, also check if prediction is contained within GT
            # This handles cases like "Antalya" in "Antalya Security Directorate Building"
            if gt["label"] == p["label"]:
                # Check if prediction is contained within ground truth (allow small boundary differences)
                pred_contained = (gt["start"] <= p["start"] + 1 and p["end"] <= gt["end"] + 1)
                # Or check if there's significant overlap (>= 50% of prediction)
                overlap_ratio = max(0, min(p["end"], gt["end"]) - max(p["start"], gt["start"])) / (p["end"] - p["start"])
                
                if pred_contained or overlap_ratio >= 0.5 or inter >= threshold:
                    matched_gts.append(gi)
            else:
                # Different labels, use regular IoU threshold
                if inter >= threshold:
                    matched_gts.append(gi)
        
        # If no normal match found, check for special LOC→ORG matches
        if not matched_gts and check_special_loc_org_match(p["label"], p["start"], p["end"], filtered_annotations, text):
            # This is a special LOC→ORG match - count as true positive
            p["used"] = True
            tp.append({
                "start": p["start"],
                "end": p["end"],
                "label": "LOC→ORG",  # Mark as special match
                "ground_truth_text": f"LOC within ORG/DEM entity",
                "predicted_text": p["pred_text"]
            })
            continue
            
        if matched_gts:
            p["used"] = True
            for gi in matched_gts:
                gt_matched[gi] = True
                tp.append({
                    "start": gt_list[gi]["start"],
                    "end": gt_list[gi]["end"],
                    "label": gt_list[gi]["label"],
                    "ground_truth_text": gt_list[gi]["gt_text"],
                    "predicted_text": p["pred_text"]
                })

    # False negatives: ground truth entities not matched by any prediction
    for gi, gt in enumerate(gt_list):
        if not gt_matched[gi] and gt["label"] not in IGNORE_FN_LABELS:
            fn.append({
                "start": gt["start"],
                "end": gt["end"],
                "label": gt["label"],
                "ground_truth_text": gt["gt_text"]
            })

    # False positives: predictions not matched to any ground truth
    # But exclude predictions that significantly overlap with already-matched ground truth entities
    for p in pred_list:
        if not p["used"] and p["label"] not in IGNORE_FP_LABELS:
            # Check if this prediction significantly overlaps with any matched ground truth
            is_overlapping_with_matched_gt = False
            for gi, gt in enumerate(gt_list):
                if gt_matched[gi]:  # Only check already matched ground truth entities
                    # Calculate overlap ratio
                    overlap = max(0, min(p["end"], gt["end"]) - max(p["start"], gt["start"]))
                    pred_length = p["end"] - p["start"]
                    overlap_ratio = overlap / pred_length if pred_length > 0 else 0
                    
                    # If prediction overlaps significantly (>= 50%) with a matched GT, don't count as FP
                    if overlap_ratio >= 0.5:
                        is_overlapping_with_matched_gt = True
                        break
            
            if not is_overlapping_with_matched_gt:
                fp.append({
                    "start": p["start"],
                    "end": p["end"],
                    "label": p["label"],
                    "predicted_text": p["pred_text"]
                })

    return {"true_positives": tp, "false_negatives": fn, "false_positives": fp}

def aggregate_results(all_docs):
    total_tp = sum(len(d["true_positives"]) for d in all_docs)
    total_fn = sum(len(d["false_negatives"]) for d in all_docs)
    total_fp = sum(len(d["false_positives"]) for d in all_docs)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"true_positives": total_tp, "false_negatives": total_fn, "false_positives": total_fp,
            "precision": precision, "recall": recall, "f1": f1}

def main(args):
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats: direct list or nested under "documents" key
    if isinstance(data, dict) and "documents" in data:
        documents = data["documents"]
    elif isinstance(data, list):
        documents = data
    else:
        print("Error: Unexpected data format in input file")
        return

    detailed = []
    for doc in documents:
        eval_doc = evaluate_document(doc, iou_threshold=args.iou_threshold)
        detailed.append({"doc_id": doc.get("doc_id"), "evaluation": eval_doc})

    agg = aggregate_results([d["evaluation"] for d in detailed])

    print("===== Piranha Evaluation (with LOC↔ORG relaxed matching) =====")
    print(f"Documents evaluated : {len(detailed)}")
    print(f"TP : {agg['true_positives']}")
    print(f"FP : {agg['false_positives']}")
    print(f"FN : {agg['false_negatives']}")
    print(f"Precision : {agg['precision']:.4f}")
    print(f"Recall    : {agg['recall']:.4f}")
    print(f"F1        : {agg['f1']:.4f}")
    print(f"Note: LOC predictions within ORG entities are credited as matches")

    if args.out_detailed:
        with open(args.out_detailed, "w", encoding="utf-8") as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)
        print(f"Detailed per-document evaluation saved to: {args.out_detailed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Piranha output")
    parser.add_argument("--input", type=str, default="output_piranha_eval.json")
    parser.add_argument("--out_detailed", type=str, default="detailed_piranha_eval.json")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
