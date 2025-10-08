#!/usr/bin/env python3
"""
evaluate_presidio_relaxed_org_loc.py

Usage:
    python evaluate_presidio_relaxed_org_loc.py \
        --input output_presidio_eval.json \
        --out_detailed detailed_eval_relaxed.json \
        --iou_threshold 0.5

This script implements RELAXED ORG-LOC matching:
- If ground truth has ORG and Presidio finds LOC (or vice versa), consider it a TRUE POSITIVE
- All other matching rules remain the same
- This addresses the common ORG/LOC semantic overlap issue in geographical entity recognition
"""

import json
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict

# Map Presidio labels -> your ground truth labels
ENTITY_MAPPING = {
    "DATE_TIME": "DATETIME",
    "NRP": "DEM",
    "PERSON": "PERSON",
    "LOC": "LOC",
    "LOCATION": "LOC",  # Presidio uses 'LOCATION', GT uses 'LOC'
    # extend this mapping as needed
}

# Labels to IGNORE when counting False Positives AND False Negatives (Presidio doesn't detect these)
IGNORE_FP_LABELS = {"CODE", "ORG", "QUANTITY", "MISC"}

# NEW: ORG-LOC equivalence groups for relaxed matching
ORG_LOC_EQUIVALENCE = {"ORG", "LOC"}

def normalize_label(presidio_label: str) -> str:
    return ENTITY_MAPPING.get(presidio_label, presidio_label)

def are_labels_compatible(gt_label: str, pred_label: str) -> bool:
    """
    Check if two labels are compatible for matching.
    Returns True if:
    1. They are exactly the same, OR
    2. Both are in the ORG-LOC equivalence group (relaxed matching)
    """
    if gt_label == pred_label:
        return True
    
    # Relaxed ORG-LOC matching
    if gt_label in ORG_LOC_EQUIVALENCE and pred_label in ORG_LOC_EQUIVALENCE:
        return True
    
    return False

def iou_span(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Compute IoU (intersection over union) for two character spans."""
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(0, inter_end - inter_start)
    union = (a_end - a_start) + (b_end - b_start) - intersection
    if union == 0:
        return 0.0
    return intersection / union

def evaluate_document_cumulative_iou(doc, iou_threshold=0.3):
    text = doc.get("text", "")
    gt_list = [
        {
            "start": int(a["start_offset"]),
            "end": int(a["end_offset"]),
            "label": a["entity_type"],
            "gt_text": text[int(a["start_offset"]):int(a["end_offset"])]
        }
        for a in doc.get("annotations", [])
    ]
    pred_list = [
        {
            "start": int(p["start"]),
            "end": int(p["end"]),
            "label": ENTITY_MAPPING.get(p["label"], p["label"]),
            "pred_text": p.get("text", text[int(p["start"]):int(p["end"])]),
            "used": False
        }
        for p in doc.get("presidio_detected_pii", [])
    ]

    tp_list, fn_list, fp_list = [], [], []

    for gt in gt_list:
        gt_len = gt["end"] - gt["start"]
        if gt_len <= 0:
            continue
        total_intersection = 0
        contributing_preds = []

        for pred in pred_list:
            if pred["used"]:
                continue
            
            # NEW: Use relaxed label matching instead of exact match
            if not are_labels_compatible(gt["label"], pred["label"]):
                continue
                
            inter_start = max(gt["start"], pred["start"])
            inter_end = min(gt["end"], pred["end"])
            intersection = max(0, inter_end - inter_start)
            if intersection > 0:
                total_intersection += intersection
                contributing_preds.append(pred)

        coverage = total_intersection / gt_len

        if coverage >= iou_threshold and contributing_preds:
            # TP
            for p in contributing_preds:
                p["used"] = True  # mark prediction as used
            
            # Show if this was a relaxed match
            relaxed_match = any(gt["label"] != p["label"] for p in contributing_preds)
            match_type = "relaxed_org_loc" if relaxed_match else "exact"
            
            tp_list.append({
                "start": gt["start"],
                "end": gt["end"],
                "label": gt["label"],
                "ground_truth_text": gt["gt_text"],
                "presidio_spanned_text": " + ".join([p["pred_text"] for p in contributing_preds]),
                "presidio_labels": [p["label"] for p in contributing_preds],
                "coverage": coverage,
                "match_type": match_type
            })
        else:
            # FN - but only if not in ignored labels
            if gt["label"] not in IGNORE_FP_LABELS:
                fn_list.append({
                    "start": gt["start"],
                    "end": gt["end"],
                    "label": gt["label"],
                    "ground_truth_text": gt["gt_text"]
                })

    # Remaining predictions = FP
    for pred in pred_list:
        if not pred["used"] and pred["label"] not in IGNORE_FP_LABELS:
            fp_list.append({
                "start": pred["start"],
                "end": pred["end"],
                "label": pred["label"],
                "presidio_spanned_text": pred["pred_text"]
            })

    return {
        "true_positives": tp_list,
        "false_negatives": fn_list,
        "false_positives": fp_list
    }

def aggregate_results(all_doc_results: List[dict]):
    total_tp = sum(len(d["true_positives"]) for d in all_doc_results)
    total_fp = sum(len(d["false_positives"]) for d in all_doc_results)
    total_fn = sum(len(d["false_negatives"]) for d in all_doc_results)

    # Count relaxed matches
    total_relaxed_matches = 0
    for doc_result in all_doc_results:
        for tp in doc_result["true_positives"]:
            if tp.get("match_type") == "relaxed_org_loc":
                total_relaxed_matches += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "relaxed_org_loc_matches": total_relaxed_matches,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main(args):
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    detailed_per_doc = []
    for doc in data:
        res = evaluate_document_cumulative_iou(doc, iou_threshold=args.iou_threshold)
        detailed_per_doc.append({
            "doc_id": doc.get("doc_id"),
            "evaluation": res
        })

    agg = aggregate_results([d["evaluation"] for d in detailed_per_doc])

    # Print summary
    print("===== Evaluation summary (RELAXED ORG-LOC MATCHING) =====")
    print(f"Documents evaluated        : {len(detailed_per_doc)}")
    print(f"True Positives             : {agg['true_positives']}")
    print(f"  - Relaxed ORG-LOC matches : {agg['relaxed_org_loc_matches']}")
    print(f"False Positives            : {agg['false_positives']} (ignoring: {IGNORE_FP_LABELS})")
    print(f"False Negatives            : {agg['false_negatives']}")
    print(f"Precision                  : {agg['precision']:.4f}")
    print(f"Recall                     : {agg['recall']:.4f}")
    print(f"F1 score                   : {agg['f1']:.4f}")
    print(f"\nRelaxed matching strategy: ORG ↔ LOC are considered equivalent")

    if args.out_detailed:
        with open(args.out_detailed, "w", encoding="utf-8") as f:
            json.dump(detailed_per_doc, f, indent=2, ensure_ascii=False)
        print(f"Detailed per-document eval saved to: {args.out_detailed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Presidio output with relaxed ORG-LOC matching.")
    parser.add_argument("--input", type=str, default="output_presidio_eval.json",
                        help="Path to the Presidio output JSON (with annotations and presidio_detected_pii).")
    parser.add_argument("--out_detailed", type=str, default="detailed_eval_relaxed.json",
                        help="Optional path to save per-document detailed evaluation JSON.")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for considering a GT and Pred span a match (default 0.5).")
    args = parser.parse_args()

    main(args)