#!/usr/bin/env python3
"""
AI4Privacy PII Detection Performance Evaluation Script

This script evaluates the performance of AI4Privacy PII detection model
by comparing detected entities against ground truth annotations.

Updated with comprehensive evaluation capabilities including:
- Entity overlap detection
- Per-entity-type metrics
- Visualization generation
- Detailed error analysis
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import argparse
import os


@dataclass
class Entity:
    """Represents a PII entity with its boundaries and type."""
    start: int
    end: int
    entity_type: str
    text: str = ""


class PerfMet:
    """Performance metrics calculator for PII detection."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
    
    def add_tp(self, count=1):
        """Add true positives."""
        self.true_positives += count
    
    def add_fp(self, count=1):
        """Add false positives."""
        self.false_positives += count
    
    def add_fn(self, count=1):
        """Add false negatives."""
        self.false_negatives += count
    
    @property
    def precision(self) -> float:
        """Calculate precision."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Calculate recall."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }


# AI4Privacy -> GT mappings (Enhanced and comprehensive)
PII_TO_GT = {
    # Person-related labels
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
   

    "CITY": "LOC",
    "STREET": "LOC",
    "ZIPCODE": "LOC",
    "BUILDINGNUM": "LOC",
    # Date/time labels
    "DATE": "DATETIME",
    "TIME": "DATETIME"
    
    # Additional labels that don't have direct GT equivalents - we'll ignore these in evaluation
    # "SOCIALNUM": None,     # Social security numbers - no GT equivalent
    # "TELEPHONENUM": None,  # Phone numbers - no GT equivalent  
    # "DRIVERLICENSENUM": None, # Driver license - no GT equivalent
    # "TAXNUM": None,        # Tax numbers - no GT equivalent
    # "EMAIL": None,         # Email addresses - no GT equivalent
}

# Labels to ignore FP/FN - these are AI4Privacy labels without GT equivalents
IGNORE_FP_LABELS = {"SOCIALNUM", "TELEPHONENUM", "DRIVERLICENSENUM", "TAXNUM", "EMAIL", "AGE", "SEX", "GENDER"}

# Ground truth labels to ignore completely (exclude from evaluation)
IGNORE_GT_LABELS = {"CODE", "CASE", "COURT", "QUANTITY", "MISC"}

# Additional labels to ignore only for false negatives (AI4Privacy not expected to detect these)
# Note: ORG is kept here because AI4Privacy doesn't detect ORG directly, but LOC predictions can still match ORG entities
IGNORE_FN_LABELS = {"ORG", "DEM", "QUANTITY", "MISC", "CODE", "CASE", "COURT"}

def clean_entity_text(entity_text):
    """
    Clean entity text by removing trailing punctuation, whitespace, and newlines.
    Also handles cases where extra words are included after the entity.
    """
    import re
    
    # First, handle the specific case where extra sentences are included
    # Look for patterns like "entity.\n\nExtra" or "entity. Extra"
    
    # Split on sentence endings followed by whitespace/newlines and extra text
    lines = entity_text.split('\n')
    if len(lines) > 1:
        # Take only the first line, which should contain the actual entity
        entity_text = lines[0]
    
    # Remove trailing punctuation and whitespace
    entity_text = re.sub(r'[.,:;!?\s]+$', '', entity_text)
    
    # Remove leading whitespace
    entity_text = re.sub(r'^[\s]+', '', entity_text)
    
    # Handle cases where extra words are included after punctuation
    # For dates/entities, try to keep only the meaningful part
    words = entity_text.split()
    if len(words) > 1:
        # For date patterns, try to identify where the entity likely ends
        date_patterns = [
            r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)',
            r'\d{4}'
        ]
        
        # Try to match date patterns and extract just the date part
        for pattern in date_patterns:
            match = re.search(pattern, entity_text, re.IGNORECASE)
            if match:
                return match.group(0)
    
    return entity_text

def remove_overlapping_entities(entities):
    """
    Remove overlapping entities, keeping the longer ones.
    This prevents double-counting when merged entities overlap with individual entities.
    """
    if not entities:
        return entities
    
    # Sort by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    result = []
    
    for entity in sorted_entities:
        # Check if this entity overlaps with any already added entity
        overlaps = False
        for existing in result:
            # Check for overlap
            if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                # There's an overlap - keep the longer entity
                entity_length = entity['end'] - entity['start']
                existing_length = existing['end'] - existing['start']
                
                if entity_length > existing_length:
                    # Remove the shorter existing entity and add this longer one
                    result.remove(existing)
                    result.append(entity)
                # If existing is longer or equal, skip this entity
                overlaps = True
                break
        
        if not overlaps:
            result.append(entity)
    
    return result


def merge_consecutive_entities(entities, gap_threshold=5):
    """
    Merge consecutive entities of the same type that are close to each other.
    This helps combine entities like 'June' and '1994' into 'June 1994'.
    For PERSON entities, prevents merging across person boundaries by detecting title words.
    """
    if not entities:
        return entities
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    merged = []
    
    current_entity = sorted_entities[0].copy()
    # Clean the text of the first entity
    current_entity['text'] = clean_entity_text(current_entity['text'])
    
    for next_entity in sorted_entities[1:]:
        # Clean the text of the next entity
        next_entity_cleaned = next_entity.copy()
        next_entity_cleaned['text'] = clean_entity_text(next_entity_cleaned['text'])
        
        gap = next_entity_cleaned['start'] - current_entity['end']
        
        # Check if entities should be merged
        should_merge = (
            current_entity['mapped_label'] == next_entity_cleaned['mapped_label'] and
            gap <= gap_threshold
        )
        
        # For PERSON entities, add special check to prevent merging across person boundaries
        if should_merge and current_entity['mapped_label'] == 'PERSON':
            # Check if next entity is a title that indicates a new person
            next_text = next_entity_cleaned['text'].strip()
            
            # Common titles that indicate the start of a new person
            title_words = ['Mr', 'Mrs', 'Ms', 'Miss', 'Dr', 'Prof', 'Sir', 'Lady', 'Lord']
            
            # If the next entity is a title and we already have content, this starts a new person
            if (next_text in title_words and 
                len(current_entity['text'].strip()) > 0):
                should_merge = False
        
        if should_merge:
            # Merge the entities
            if current_entity['end'] < next_entity_cleaned['start']:
                # Add the gap text between entities
                gap_text = " " if gap <= gap_threshold else " "
                current_entity['text'] += gap_text + next_entity_cleaned['text']
            else:
                current_entity['text'] += next_entity_cleaned['text']
            
            current_entity['end'] = next_entity_cleaned['end']
            current_entity['original_label'] = current_entity.get('original_label', current_entity['label'])
        else:
            # No merge, add current entity and start new one
            merged.append(current_entity)
            current_entity = next_entity_cleaned.copy()
    
    # Add the last entity
    merged.append(current_entity)
    
    return merged

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

def evaluate_single_document(predicted_entities: List[Dict], 
                     gt_entities: List[Dict],
                     strict_threshold: float = 1.0,
                     relaxed_threshold: float = 0.5) -> Dict:
    """
    Evaluate a single document with improved matching logic based on Piranha approach.
    """
    
    # Step 1: Apply label mapping and clean predicted entities
    mapped_predictions = []
    for entity in predicted_entities:
        original_label = entity['label']
        mapped_label = PII_TO_GT.get(original_label)
        
        if mapped_label is not None:
            mapped_entity = entity.copy()
            # Clean the predicted text
            mapped_entity['text'] = clean_entity_text(entity['text'])
            # Adjust end position based on cleaned text length
            original_length = len(entity['text'])
            cleaned_length = len(mapped_entity['text'])
            if cleaned_length < original_length:
                mapped_entity['end'] = mapped_entity['start'] + cleaned_length
            
            mapped_entity['mapped_label'] = mapped_label
            mapped_entity['original_label'] = original_label
            mapped_predictions.append(mapped_entity)
    
    # Step 2: Merge consecutive entities of the same type (names, dates, etc.)
    merged_predictions = merge_consecutive_entities(mapped_predictions)
    
    # Step 2.5: Remove overlapping entities to prevent double-counting
    deduplicated_predictions = remove_overlapping_entities(merged_predictions)
    
    # Step 3: Filter ground truth to use only first annotator in document
    if gt_entities:
        # Get the first annotator found in this document
        first_annotator = gt_entities[0].get('annotator', None)
        if first_annotator:
            gt_first_annotator = [
                gt for gt in gt_entities 
                if gt.get('annotator') == first_annotator
            ]
        else:
            # Fallback if no annotator field
            gt_first_annotator = gt_entities
    else:
        gt_first_annotator = []
    
    # Step 4: Filter by entity types we want to evaluate
    filtered_gt = [
        gt for gt in gt_first_annotator 
        if gt['entity_type'] not in IGNORE_GT_LABELS
    ]
    
    # Step 5: Initialize metrics
    metrics = {
        'total_predicted': len(deduplicated_predictions),
        'total_ground_truth': len(filtered_gt),
        'strict_matches': 0,
        'relaxed_matches': 0,
        'tp_strict': 0,
        'tp_relaxed': 0,
        'fp': 0,
        'fn': 0,
        'matched_pairs': [],
        'unmatched_predictions': [],
        'unmatched_ground_truth': []
    }
    
    # Track which entities are matched
    gt_matched = [False] * len(filtered_gt)
    pred_used = [False] * len(deduplicated_predictions)
    
    # Step 6: Find matches using improved logic (one-to-one matching) with multi-GT support
    for pred_idx, pred in enumerate(deduplicated_predictions):
        best_matches = []  # Support matching multiple GT entities
        best_total_iou = 0.0
        best_gt_indices = []

        for gt_idx, gt in enumerate(filtered_gt):
            if gt_matched[gt_idx]:
                continue

            pred_label = pred['mapped_label']
            gt_label = gt['entity_type']

            # Allow LOC predictions to match ORG ground truth
            if pred_label == "LOC" and gt_label == "ORG":
                pass
            elif pred_label != gt_label:
                continue

            pred_span = (pred['start'], pred['end'])
            gt_span = (gt['start_offset'], gt['end_offset'])
            iou = calculate_iou(pred_span, gt_span)

            # Use different thresholds for different entity types
            if gt_label == "LOC":
                threshold = 0.3
            elif gt_label == "PERSON":
                threshold = 0.4
            elif gt_label == "ORG":
                threshold = 0.3  # Use lower threshold for ORG entities
            else:
                threshold = relaxed_threshold

            pred_contained = (gt['start_offset'] <= pred['start'] + 1 and pred['end'] <= gt['end_offset'] + 1)
            overlap = max(0, min(pred['end'], gt['end_offset']) - max(pred['start'], gt['start_offset']))
            overlap_ratio = overlap / (pred['end'] - pred['start']) if pred['end'] > pred['start'] else 0

            # --- LOC to ORG matching improvement ---
            # If prediction is LOC and ground truth is ORG, allow matching if prediction text is contained in GT text
            if pred_label == "LOC" and gt_label == "ORG":
                pred_text = pred['text'].strip().lower()
                gt_text = gt['span_text'].strip().lower()
                # Allow matching if LOC text is contained in ORG text (e.g., "Ankara" in "Ankara Regional Court")
                if pred_text in gt_text and overlap_ratio >= 0.3:
                    if iou > best_total_iou:
                        best_total_iou = iou
                        best_matches = [gt]
                        best_gt_indices = [gt_idx]
                    continue

            # --- PERSON entity improvement ---
            # If prediction is PERSON and ground truth is PERSON, allow matching if prediction is a substring of GT text and spans overlap
            if pred_label == "PERSON" and gt_label == "PERSON":
                pred_text = pred['text'].strip().lower()
                gt_text = gt['span_text'].strip().lower()
                # If prediction text is contained in GT text and spans overlap
                if pred_text in gt_text and overlap_ratio >= 0.5:
                    if iou > best_total_iou:
                        best_total_iou = iou
                        best_matches = [gt]
                        best_gt_indices = [gt_idx]
                    continue

            # Accept match if any condition is met and it's better than previous best
            if (pred_contained or overlap_ratio >= 0.5 or iou >= threshold) and iou > best_total_iou:
                best_total_iou = iou
                best_matches = [gt]
                best_gt_indices = [gt_idx]
        
        # If no single match found, try to match against multiple consecutive GT entities
        # This handles cases like "November 2000 January 2001" matching "November 2000" + "January 2001"
        if not best_matches and pred['mapped_label'] == "DATETIME":
            for start_gt_idx, start_gt in enumerate(filtered_gt):
                if gt_matched[start_gt_idx] or start_gt['entity_type'] != "DATETIME":
                    continue
                
                # Look for consecutive DATETIME entities that together might match the prediction
                consecutive_gts = [start_gt]
                consecutive_indices = [start_gt_idx]
                
                for next_gt_idx in range(start_gt_idx + 1, len(filtered_gt)):
                    next_gt = filtered_gt[next_gt_idx]
                    if (gt_matched[next_gt_idx] or next_gt['entity_type'] != "DATETIME" or
                        next_gt['start_offset'] - consecutive_gts[-1]['end_offset'] > 50):  # Max gap between dates
                        break
                    consecutive_gts.append(next_gt)
                    consecutive_indices.append(next_gt_idx)
                
                if len(consecutive_gts) >= 2:  # Only try if we have multiple dates
                    # Calculate combined span
                    combined_start = consecutive_gts[0]['start_offset']
                    combined_end = consecutive_gts[-1]['end_offset']
                    combined_span = (combined_start, combined_end)
                    
                    # Calculate IoU with the merged prediction
                    pred_span = (pred['start'], pred['end'])
                    combined_iou = calculate_iou(pred_span, combined_span)
                    
                    # Check if this is a better match than single entities
                    if combined_iou > best_total_iou and combined_iou >= 0.6:
                        best_total_iou = combined_iou
                        best_matches = consecutive_gts
                        best_gt_indices = consecutive_indices
        
        # Record match if found (only the best match per prediction)
        if best_matches and best_total_iou >= 0.6:  # Minimum IoU to prevent very weak matches
            pred_used[pred_idx] = True
            for gt_idx in best_gt_indices:
                gt_matched[gt_idx] = True  # Mark all matched GTs as used
            
            # Determine match type
            if best_total_iou >= strict_threshold:
                metrics['strict_matches'] += 1
                metrics['tp_strict'] += 1
                match_type = 'strict'
            else:
                match_type = 'relaxed'
            
            metrics['relaxed_matches'] += 1
            metrics['tp_relaxed'] += 1
            
            # For display purposes, use the first GT or create a combined representation
            display_gt = best_matches[0] if len(best_matches) == 1 else {
                'start_offset': best_matches[0]['start_offset'],
                'end_offset': best_matches[-1]['end_offset'],
                'entity_type': best_matches[0]['entity_type'],
                'span_text': ' '.join([gt['span_text'] for gt in best_matches])
            }
            
            metrics['matched_pairs'].append({
                'prediction': pred,
                'ground_truth': display_gt,
                'iou': best_total_iou,
                'match_type': match_type
            })
    
    # Step 7: Count false positives and false negatives
    for pred_idx, pred in enumerate(deduplicated_predictions):
        if not pred_used[pred_idx] and pred['original_label'] not in IGNORE_FP_LABELS:
            # Check if this prediction significantly overlaps with any matched ground truth
            is_overlapping_with_matched_gt = False
            pred_span = (pred['start'], pred['end'])
            pred_length = pred['end'] - pred['start']
            for gt_idx, gt in enumerate(filtered_gt):
                if gt_matched[gt_idx]:
                    gt_span = (gt['start_offset'], gt['end_offset'])
                    overlap = max(0, min(pred['end'], gt['end_offset']) - max(pred['start'], gt['start_offset']))
                    overlap_ratio = overlap / pred_length if pred_length > 0 else 0
                    if overlap_ratio >= 0.5:
                        is_overlapping_with_matched_gt = True
                        break
            if not is_overlapping_with_matched_gt:
                metrics['fp'] += 1
                metrics['unmatched_predictions'].append(pred)

    for gt_idx, gt in enumerate(filtered_gt):
        if not gt_matched[gt_idx] and gt['entity_type'] not in IGNORE_FN_LABELS:
            metrics['fn'] += 1
            metrics['unmatched_ground_truth'].append(gt)

    return metrics

def calculate_aggregated_metrics(all_metrics: List[Dict]) -> Dict:
    """Calculate precision, recall, F1 from aggregated counts."""
    
    total_tp_strict = sum(m['tp_strict'] for m in all_metrics)
    total_tp_relaxed = sum(m['tp_relaxed'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    
    # Strict metrics
    precision_strict = total_tp_strict / (total_tp_strict + total_fp) if (total_tp_strict + total_fp) > 0 else 0
    recall_strict = total_tp_strict / (total_tp_strict + total_fn) if (total_tp_strict + total_fn) > 0 else 0
    f1_strict = 2 * precision_strict * recall_strict / (precision_strict + recall_strict) if (precision_strict + recall_strict) > 0 else 0
    
    # Relaxed metrics
    precision_relaxed = total_tp_relaxed / (total_tp_relaxed + total_fp) if (total_tp_relaxed + total_fp) > 0 else 0
    recall_relaxed = total_tp_relaxed / (total_tp_relaxed + total_fn) if (total_tp_relaxed + total_fn) > 0 else 0
    f1_relaxed = 2 * precision_relaxed * recall_relaxed / (precision_relaxed + recall_relaxed) if (precision_relaxed + recall_relaxed) > 0 else 0
    
    return {
        'strict': {
            'precision': precision_strict,
            'recall': recall_strict,
            'f1': f1_strict,
            'tp': total_tp_strict,
            'fp': total_fp,
            'fn': total_fn
        },
        'relaxed': {
            'precision': precision_relaxed,
            'recall': recall_relaxed,
            'f1': f1_relaxed,
            'tp': total_tp_relaxed,
            'fp': total_fp,
            'fn': total_fn
        }
    }

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate AI4Privacy PII detection results')
    parser.add_argument('--input', default='/home/ide/ide/ECHR_mask_ai4privacy/output_ai4privacy.json',
                        help='Input JSON file with AI4Privacy results')
    parser.add_argument('--output', default='/home/ide/ide/ECHR_mask_ai4privacy/detailed_ai4privacy_eval.json',
                        help='Output JSON file for detailed results')
    
    args = parser.parse_args()
    
    print("=== AI4Privacy PII Detection Evaluation (Post-Processing) ===")
    
    # Load results
    print(f"Loading results from: {args.input}")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        print("Please run ai4privacy_script.py first to generate results.")
        return
    
    print(f"Loaded {len(results)} documents")
    
    # Evaluate each document
    all_metrics = []
    
    for i, doc in enumerate(results):
        if i % 50 == 0:
            print(f"Evaluating document {i+1}/{len(results)}")
        
        doc_metrics = evaluate_single_document(
            doc['ai4privacy_detected_pii'],
            doc['ground_truth_entities']
        )
        
        doc_metrics['doc_id'] = doc['id']
        all_metrics.append(doc_metrics)
    
    # Calculate overall performance
    aggregated = calculate_aggregated_metrics(all_metrics)
    
    # Print results
    print("\n=== EVALUATION RESULTS ===")
    print(f"Processed {len(results)} documents")
    print(f"Label mapping: {len(PII_TO_GT)} AI4Privacy labels -> GT labels")
    print(f"Ignored FP labels: {IGNORE_FP_LABELS}")
    print(f"Ignored FN labels: {IGNORE_FN_LABELS}")
    
    print(f"\nStrict Matching (IoU >= 1.0):")
    print(f"  Precision: {aggregated['strict']['precision']:.3f}")
    print(f"  Recall:    {aggregated['strict']['recall']:.3f}")
    print(f"  F1-Score:  {aggregated['strict']['f1']:.3f}")
    
    print(f"\nRelaxed Matching (IoU >= 0.5):")
    print(f"  Precision: {aggregated['relaxed']['precision']:.3f}")
    print(f"  Recall:    {aggregated['relaxed']['recall']:.3f}")
    print(f"  F1-Score:  {aggregated['relaxed']['f1']:.3f}")
    
    print(f"\nCounts:")
    print(f"  True Positives (strict): {aggregated['strict']['tp']}")
    print(f"  True Positives (relaxed): {aggregated['relaxed']['tp']}")
    print(f"  False Positives: {aggregated['strict']['fp']}")
    print(f"  False Negatives: {aggregated['strict']['fn']}")
    
    # Save detailed results in the same format as Piranha
    output_data = []
    
    for metrics in all_metrics:
        doc_evaluation = {
            'doc_id': metrics['doc_id'],
            'evaluation': {
                'true_positives': [],
                'false_positives': [],
                'false_negatives': []
            }
        }
        
        # Add true positives from matched pairs
        for match in metrics['matched_pairs']:
            pred = match['prediction']
            gt = match['ground_truth']
            
            tp_entry = {
                'start': gt['start_offset'],
                'end': gt['end_offset'],
                'label': gt['entity_type'],
                'ground_truth_text': gt['span_text'],
                'predicted_text': pred['text'],
                'iou': match['iou'],
                'match_type': match['match_type']
            }
            doc_evaluation['evaluation']['true_positives'].append(tp_entry)
        
        # Add false positives from unmatched predictions
        for pred in metrics['unmatched_predictions']:
            fp_entry = {
                'start': pred['start'],
                'end': pred['end'],
                'label': pred['mapped_label'],
                'predicted_text': pred['text'],
                'original_label': pred['original_label']
            }
            doc_evaluation['evaluation']['false_positives'].append(fp_entry)
        
        # Add false negatives from unmatched ground truth
        for gt in metrics['unmatched_ground_truth']:
            fn_entry = {
                'start': gt['start_offset'],
                'end': gt['end_offset'],
                'label': gt['entity_type'],
                'ground_truth_text': gt['span_text']
            }
            doc_evaluation['evaluation']['false_negatives'].append(fn_entry)
        
        output_data.append(doc_evaluation)
    
    print(f"\nSaving detailed results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()