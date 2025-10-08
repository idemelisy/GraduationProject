#!/usr/bin/env python3
"""
piranha_detect_evaluate.py

- Loads annotations.json with ground truth in ECHR format
- Runs Piranha detection with proper offset handling
- Evaluates performance using multiple metrics (strict, relaxed, token-level)
- Handles label mismatches and entity boundary issues
- Saves detailed results and metrics
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
from typing import List, Dict, Tuple, Set

# ---- Configuration ----
model_name = "iiiorg/piiranha-v1-detect-personal-information"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping: Piranha labels -> ECHR ground truth labels
LABEL_MAPPING = {
    # Piranha labels
    'GIVENNAME': 'PERSON',
    'SURNAME': 'PERSON',
    'STREET': 'LOC',
    'CITY': 'LOC',
    'DATEOFBIRTH': 'DATETIME',
    'BUILDING': 'LOC',
    'ORGANIZATION': 'ORG',
    'ZIPCODE': 'LOC',
    'DATE': 'DATETIME',
    
    # ECHR labels (keep as-is)
    'PERSON': 'PERSON',
    'LOC': 'LOC',
    'ORG': 'ORG',
    'DATETIME': 'DATETIME',
    'DATE': 'DATETIME',
    'GPE': 'LOC',
    'LOCATION': 'LOC',
}

# Define which ECHR labels Piranha CAN reasonably detect
# Filter evaluation to only these labels for fair comparison
EVALUABLE_LABELS = {'PERSON', 'LOC', 'ORG', 'DATETIME'}

# Labels Piranha CANNOT detect (domain mismatch)
# These will be excluded from recall calculation
UNEVALUABLE_LABELS = {'CODE', 'DEM', 'MISC', 'QUANTITY'}

# ---- Load Model ----
print("Loading Piranha model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
model.to(device)
model.eval()
print(f"Model loaded on {device}")


def extract_entities_fixed(text: str, model, tokenizer, device, max_len=512) -> List[Dict]:
    """
    Extract entities with proper handling of:
    - Subword tokenization
    - Long texts (chunking)
    - Overlapping chunks (deduplication)
    
    Returns entities in format: [{"label": str, "start": int, "end": int, "text": str}, ...]
    """
    if not text.strip():
        return []
    
    entities = []
    text_len = len(text)
    stride = max_len - 50  # Overlap to catch entities at chunk boundaries
    
    for chunk_start in range(0, text_len, stride):
        chunk_end = min(chunk_start + max_len, text_len)
        chunk_text = text[chunk_start:chunk_end]
        
        # Tokenize with offset mapping
        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
        id2label = model.config.id2label
        
        # Build entities from BIO tags
        current_entity = None
        
        for token_idx, pred_id in enumerate(predictions):
            label = id2label[pred_id]
            token_start, token_end = offset_mapping[token_idx]
            
            # Skip special tokens and empty tokens
            if token_start == token_end:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Adjust offsets to original text position
            abs_start = chunk_start + token_start
            abs_end = chunk_start + token_end
            
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Extract base label (remove B-/I- prefix)
            if label.startswith("B-") or label.startswith("I-"):
                base_label = label[2:]
            else:
                base_label = label
            
            # Handle B- tags (start of entity)
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": base_label,
                    "start": abs_start,
                    "end": abs_end,
                }
            # Handle I- tags (continuation)
            elif label.startswith("I-"):
                if current_entity and current_entity["label"] == base_label:
                    # Extend current entity
                    current_entity["end"] = abs_end
                else:
                    # Orphan I- tag, treat as new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "label": base_label,
                        "start": abs_start,
                        "end": abs_end,
                    }
            # Handle tags without prefix
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": base_label,
                    "start": abs_start,
                    "end": abs_end,
                }
        
        if current_entity:
            entities.append(current_entity)
        
        # Break if we've processed the entire text
        if chunk_end >= text_len:
            break
    
    # Deduplicate overlapping entities from chunk boundaries
    entities = deduplicate_entities(entities, text)
    
    # Add text field
    for e in entities:
        e["text"] = text[e["start"]:e["end"]]
    
    return entities


def deduplicate_entities(entities: List[Dict], text: str) -> List[Dict]:
    """Remove duplicate/overlapping entities from chunked processing."""
    if not entities:
        return []
    
    # Sort by start position
    sorted_entities = sorted(entities, key=lambda x: (x["start"], -x["end"]))
    
    # Remove exact duplicates and merge overlapping same-label entities
    deduplicated = []
    prev = None
    
    for entity in sorted_entities:
        if prev is None:
            prev = entity
            continue
        
        # Exact duplicate
        if prev["start"] == entity["start"] and prev["end"] == entity["end"] and prev["label"] == entity["label"]:
            continue
        
        # Overlapping entities with same label - merge
        if (prev["label"] == entity["label"] and 
            prev["start"] <= entity["start"] < prev["end"]):
            prev["end"] = max(prev["end"], entity["end"])
            continue
        
        # No overlap - add previous and move on
        deduplicated.append(prev)
        prev = entity
    
    if prev:
        deduplicated.append(prev)
    
    return deduplicated


def normalize_label(label: str, mapping: Dict[str, str]) -> str:
    """Normalize label using mapping, return original if not in mapping."""
    label_upper = label.upper()
    return mapping.get(label_upper, label_upper)


def calculate_iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    """Calculate Intersection over Union for two spans."""
    start1, end1 = span1
    start2, end2 = span2
    
    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calculate union
    union = (end1 - start1) + (end2 - start2) - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def evaluate_entities(predicted: List[Dict], ground_truth: List[Dict], 
                      iou_threshold: float = 0.5, 
                      filter_unevaluable: bool = True) -> Dict:
    """
    Evaluate predictions against ground truth using multiple metrics.
    
    Args:
        predicted: List of dicts with keys: label, start, end
        ground_truth: List of dicts with keys: entity_type, start_offset, end_offset
        iou_threshold: IoU threshold for relaxed matching
        filter_unevaluable: If True, exclude labels Piranha cannot detect from evaluation
    
    Returns:
        - strict: exact match (boundaries + label)
        - relaxed: IoU-based match
        - partial: any overlap with correct label
        - type_only: correct label, ignore boundaries
    """
    # Filter ground truth to only evaluable labels if requested
    if filter_unevaluable:
        ground_truth_filtered = [
            gt for gt in ground_truth 
            if normalize_label(gt["label"], LABEL_MAPPING) in EVALUABLE_LABELS
        ]
        num_filtered_out = len(ground_truth) - len(ground_truth_filtered)
    else:
        ground_truth_filtered = ground_truth
        num_filtered_out = 0
    
    # Normalize labels
    pred_normalized = [
        {
            "label": e["label"],
            "normalized_label": normalize_label(e["label"], LABEL_MAPPING),
            "start": e["start"],
            "end": e["end"],
            "text": e.get("text", "")
        }
        for e in predicted
    ]
    
    gt_normalized = [
        {
            "label": e["label"],
            "normalized_label": normalize_label(e["label"], LABEL_MAPPING),
            "start": e["start"],
            "end": e["end"],
            "text": e.get("text", "")
        }
        for e in ground_truth_filtered
    ]
    
    # Track matches
    matched_pred = set()
    matched_gt = set()
    
    strict_matches = 0
    relaxed_matches = 0
    partial_matches = 0
    type_matches = 0
    
    # Store matched pairs for analysis
    match_details = []
    
    # For each predicted entity, find best matching ground truth
    for pred_idx, pred in enumerate(pred_normalized):
        best_match = None
        best_iou = 0.0
        
        for gt_idx, gt in enumerate(gt_normalized):
            if gt_idx in matched_gt:
                continue
            
            # Calculate IoU
            iou = calculate_iou(
                (pred["start"], pred["end"]),
                (gt["start"], gt["end"])
            )
            
            if iou > best_iou:
                best_iou = iou
                best_match = gt_idx
        
        if best_match is not None:
            gt = gt_normalized[best_match]
            
            # Check label match
            labels_match = pred["normalized_label"] == gt["normalized_label"]
            
            match_detail = {
                "pred": pred,
                "gt": gt,
                "iou": best_iou,
                "labels_match": labels_match
            }
            
            # Strict match: exact boundaries and label
            if (pred["start"] == gt["start"] and 
                pred["end"] == gt["end"] and 
                labels_match):
                strict_matches += 1
                matched_pred.add(pred_idx)
                matched_gt.add(best_match)
                match_detail["match_type"] = "strict"
                match_details.append(match_detail)
            
            # Relaxed match: IoU threshold and label
            elif best_iou >= iou_threshold and labels_match:
                relaxed_matches += 1
                matched_pred.add(pred_idx)
                matched_gt.add(best_match)
                match_detail["match_type"] = "relaxed"
                match_details.append(match_detail)
            
            # Partial match: any overlap and label
            elif best_iou > 0 and labels_match:
                partial_matches += 1
                matched_pred.add(pred_idx)
                matched_gt.add(best_match)
                match_detail["match_type"] = "partial"
                match_details.append(match_detail)
            
            # Type match: correct label, wrong boundaries
            elif labels_match:
                type_matches += 1
                match_detail["match_type"] = "type_only"
    
    # Calculate metrics
    num_pred = len(pred_normalized)
    num_gt = len(gt_normalized)
    
    def calc_metrics(matches, pred_count, gt_count):
        precision = matches / pred_count if pred_count > 0 else 0.0
        recall = matches / gt_count if gt_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
    
    return {
        "strict": calc_metrics(strict_matches, num_pred, num_gt),
        "relaxed": calc_metrics(relaxed_matches, num_pred, num_gt),
        "partial": calc_metrics(partial_matches, num_pred, num_gt),
        "type_only": calc_metrics(type_matches, num_pred, num_gt),
        "counts": {
            "predicted": num_pred,
            "ground_truth": num_gt,
            "ground_truth_filtered_out": num_filtered_out,
            "strict_matches": strict_matches,
            "relaxed_matches": relaxed_matches,
            "partial_matches": partial_matches,
            "type_matches": type_matches,
            "false_positives": num_pred - len(matched_pred),
            "false_negatives": num_gt - len(matched_gt)
        },
        "match_details": match_details[:10]  # Save first 10 for inspection
    }


def mask_text(text: str, entities: List[Dict]) -> str:
    """Replace detected entities with [LABEL]."""
    if not entities:
        return text
    
    # Sort by start position in reverse to avoid offset issues
    sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    masked = text
    for e in sorted_entities:
        masked = masked[:e["start"]] + f"[{e['label']}]" + masked[e["end"]:]
    
    return masked


def convert_annotations_to_entities(annotations: List[Dict]) -> List[Dict]:
    """
    Convert ECHR annotation format to standard entity format.
    
    Input: [{"entity_type": "PERSON", "start_offset": 0, "end_offset": 10, ...}, ...]
    Output: [{"label": "PERSON", "start": 0, "end": 10, ...}, ...]
    """
    return [
        {
            "label": ann["entity_type"],
            "start": ann["start_offset"],
            "end": ann["end_offset"],
            "text": ann.get("span_text", "")
        }
        for ann in annotations
    ]


def process_dataset(input_file: str, output_file: str):
    """Process entire dataset and evaluate."""
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} documents...")
    
    # Aggregate metrics
    all_metrics = []
    error_cases = []
    label_confusion = defaultdict(lambda: defaultdict(int))  # Track label mismatches
    
    for doc_idx, doc in enumerate(data):
        if doc_idx % 10 == 0:
            print(f"Processing document {doc_idx + 1}/{len(data)}...")
        
        doc_id = doc.get("doc_id", f"doc_{doc_idx}")
        text = doc.get("text", "")
        annotations = doc.get("annotations", [])
        
        if not text.strip():
            doc["piranha_masked_text"] = ""
            doc["piranha_detected_pii"] = []
            doc["evaluation"] = None
            continue
        
        # Convert annotations to entity format
        ground_truth = convert_annotations_to_entities(annotations)
        
        # Detect entities with Piranha
        predicted_entities = extract_entities_fixed(text, model, tokenizer, device)
        
        # Evaluate with filtering (fair comparison)
        metrics_filtered = evaluate_entities(predicted_entities, ground_truth, filter_unevaluable=True)
        
        # Also evaluate without filtering (full comparison)
        metrics_full = evaluate_entities(predicted_entities, ground_truth, filter_unevaluable=False)
        
        # Track label confusion
        for match in metrics_filtered.get("match_details", []):
            if not match["labels_match"]:
                pred_label = match["pred"]["normalized_label"]
                gt_label = match["gt"]["normalized_label"]
                label_confusion[pred_label][gt_label] += 1
        
        # Mask text
        masked_text = mask_text(text, predicted_entities)
        
        # Store results
        doc["piranha_masked_text"] = masked_text
        doc["piranha_detected_pii"] = predicted_entities
        doc["evaluation_filtered"] = metrics_filtered  # Fair comparison
        doc["evaluation_full"] = metrics_full  # Full comparison
        
        all_metrics.append(metrics_filtered)
        
        # Track poor performance cases for analysis
        if metrics_filtered["relaxed"]["f1"] < 0.5 and len(ground_truth) > 0:
            error_cases.append({
                "doc_id": doc_id,
                "text_preview": text[:300] + "..." if len(text) > 300 else text,
                "metrics": {
                    "strict_f1": metrics_filtered["strict"]["f1"],
                    "relaxed_f1": metrics_filtered["relaxed"]["f1"],
                },
                "predicted_count": len(predicted_entities),
                "gt_count": len(ground_truth),
                "false_positives": metrics_filtered["counts"]["false_positives"],
                "false_negatives": metrics_filtered["counts"]["false_negatives"],
                "sample_predictions": predicted_entities[:5],
                "sample_ground_truth": ground_truth[:5]
            })
    
    # Calculate aggregate metrics
    aggregate = aggregate_metrics(all_metrics)
    
    # Calculate per-label metrics
    per_label_metrics = calculate_per_label_metrics(data)
    
    # Save results
    output_data = {
        "documents": data,
        "aggregate_metrics": aggregate,
        "per_label_metrics": per_label_metrics,
        "label_confusion_matrix": dict(label_confusion),
        "error_cases_sample": error_cases[:30],  # Save first 30 error cases
        "label_mapping_used": LABEL_MAPPING,
        "evaluation_config": {
            "iou_threshold": 0.5,
            "model": model_name
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {output_file}")
    print("\n" + "="*70)
    print("FILTERED EVALUATION (Only labels Piranha can detect)")
    print(f"Evaluating only: {EVALUABLE_LABELS}")
    print(f"Excluded labels: {UNEVALUABLE_LABELS}")
    print("="*70)
    print(f"\nStrict Matching (exact boundaries + label):")
    print(f"  Precision: {aggregate['strict']['precision']:.3f}")
    print(f"  Recall:    {aggregate['strict']['recall']:.3f}")
    print(f"  F1:        {aggregate['strict']['f1']:.3f}")
    print(f"\nRelaxed Matching (IoU ≥ 0.5 + label):")
    print(f"  Precision: {aggregate['relaxed']['precision']:.3f}")
    print(f"  Recall:    {aggregate['relaxed']['recall']:.3f}")
    print(f"  F1:        {aggregate['relaxed']['f1']:.3f}")
    print(f"\nPartial Matching (any overlap + label):")
    print(f"  Precision: {aggregate['partial']['precision']:.3f}")
    print(f"  Recall:    {aggregate['partial']['recall']:.3f}")
    print(f"  F1:        {aggregate['partial']['f1']:.3f}")
    print(f"\nCounts:")
    print(f"  Total Predicted: {aggregate['total_counts']['predicted']}")
    print(f"  Ground Truth (Evaluable): {aggregate['total_counts']['ground_truth']}")
    print(f"  Ground Truth (Filtered Out): {aggregate['total_counts'].get('ground_truth_filtered_out', 0)}")
    print(f"  False Positives: {aggregate['total_counts']['false_positives']}")
    print(f"  False Negatives: {aggregate['total_counts']['false_negatives']}")
    
    if per_label_metrics:
        print(f"\n" + "="*70)
        print("PER-LABEL METRICS (Relaxed Matching, Filtered)")
        print("="*70)
        for label, metrics in sorted(per_label_metrics.items()):
            if label in EVALUABLE_LABELS:
                print(f"\n{label}:")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall:    {metrics['recall']:.3f}")
                print(f"  F1:        {metrics['f1']:.3f}")
                print(f"  Count (GT): {metrics['count']}")
    
    print("\n" + "="*70)


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """Calculate aggregate metrics across all documents."""
    if not metrics_list:
        return {}
    
    # Sum up all counts
    total_pred = sum(m["counts"]["predicted"] for m in metrics_list)
    total_gt = sum(m["counts"]["ground_truth"] for m in metrics_list)
    total_strict = sum(m["counts"]["strict_matches"] for m in metrics_list)
    total_relaxed = sum(m["counts"]["relaxed_matches"] for m in metrics_list)
    total_partial = sum(m["counts"]["partial_matches"] for m in metrics_list)
    total_type = sum(m["counts"]["type_matches"] for m in metrics_list)
    total_fp = sum(m["counts"]["false_positives"] for m in metrics_list)
    total_fn = sum(m["counts"]["false_negatives"] for m in metrics_list)
    
    def calc_agg_metrics(matches, pred, gt):
        precision = matches / pred if pred > 0 else 0.0
        recall = matches / gt if gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
    
    return {
        "strict": calc_agg_metrics(total_strict, total_pred, total_gt),
        "relaxed": calc_agg_metrics(total_relaxed, total_pred, total_gt),
        "partial": calc_agg_metrics(total_partial, total_pred, total_gt),
        "type_only": calc_agg_metrics(total_type, total_pred, total_gt),
        "total_counts": {
            "predicted": total_pred,
            "ground_truth": total_gt,
            "strict_matches": total_strict,
            "relaxed_matches": total_relaxed,
            "partial_matches": total_partial,
            "type_matches": total_type,
            "false_positives": total_fp,
            "false_negatives": total_fn
        }
    }


def calculate_per_label_metrics(documents: List[Dict]) -> Dict:
    """Calculate metrics per entity label."""
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0})
    
    for doc in documents:
        eval_data = doc.get("evaluation")
        if not eval_data:
            continue
        
        annotations = doc.get("annotations", [])
        predictions = doc.get("piranha_detected_pii", [])
        
        # Count ground truth by label
        for ann in annotations:
            label = normalize_label(ann["entity_type"], LABEL_MAPPING)
            label_stats[label]["count"] += 1
        
        # Use relaxed matches for per-label stats
        match_details = eval_data.get("match_details", [])
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        for match in match_details:
            if match.get("match_type") in ["strict", "relaxed", "partial"]:
                gt_label = match["gt"]["normalized_label"]
                label_stats[gt_label]["tp"] += 1
                matched_gt_indices.add((match["gt"]["start"], match["gt"]["end"]))
                matched_pred_indices.add((match["pred"]["start"], match["pred"]["end"]))
        
        # Count false positives
        for pred in predictions:
            if (pred["start"], pred["end"]) not in matched_pred_indices:
                label = normalize_label(pred["label"], LABEL_MAPPING)
                label_stats[label]["fp"] += 1
        
        # Count false negatives
        for ann in annotations:
            if (ann["start_offset"], ann["end_offset"]) not in matched_gt_indices:
                label = normalize_label(ann["entity_type"], LABEL_MAPPING)
                label_stats[label]["fn"] += 1
    
    # Calculate metrics per label
    per_label_metrics = {}
    for label, stats in label_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        per_label_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": stats["count"],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
    
    return per_label_metrics


if __name__ == "__main__":
    input_json_path = "/home/ide/ide/ECHR_mask/annotations.json"
    output_json_path = "/home/ide/ide/ECHR_mask/output_piranha_eval.json"
    
    process_dataset(input_json_path, output_json_path)
    print("\n✅ Evaluation complete!")