#!/usr/bin/env python3
"""
Debug and fix the DATETIME matching issue
"""

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai4privacy_evaluate import evaluate_single_document

def debug_full_evaluation():
    # Load AI4Privacy output
    with open('output_ai4privacy.json', 'r') as f:
        ai4privacy_data = json.load(f)
    
    # Find document 001-84741
    for doc in ai4privacy_data:
        if doc['id'] == '001-84741':
            print(f'Full evaluation debug for document {doc["id"]}')
            
            # Run the full evaluation with debug info
            predicted_entities = doc['ai4privacy_detected_pii']
            gt_entities = doc['ground_truth_entities']
            
            print(f'Total predicted entities: {len(predicted_entities)}')
            print(f'Total ground truth entities: {len(gt_entities)}')
            
            # Filter by first annotator
            if gt_entities:
                first_annotator = gt_entities[0].get('annotator', None)
                gt_first_annotator = [gt for gt in gt_entities if gt.get('annotator') == first_annotator]
                print(f'GT entities from first annotator: {len(gt_first_annotator)}')
            
            # Run evaluation
            result = evaluate_single_document(predicted_entities, gt_entities)
            
            print(f'\nEvaluation results:')
            print(f'  Strict TP: {result["tp_strict"]}')
            print(f'  Relaxed TP: {result["tp_relaxed"]}')
            print(f'  False Positives: {result["fp"]}')
            print(f'  False Negatives: {result["fn"]}')
            
            # Check the matched pairs
            print(f'\nMatched pairs ({len(result["matched_pairs"])}):')
            for pair in result['matched_pairs'][:10]:  # Show first 10
                pred = pair['prediction']
                gt = pair['ground_truth']
                pred_text = pred["text"]
                pred_start = pred["start"]
                pred_end = pred["end"]
                gt_text = gt["span_text"]
                gt_start = gt["start_offset"]
                gt_end = gt["end_offset"]
                iou = pair["iou"]
                print(f'  {pred_text} ({pred_start}-{pred_end}) <-> {gt_text} ({gt_start}-{gt_end}) IoU: {iou:.3f}')
            
            # Check unmatched predictions (false positives)
            print(f'\nUnmatched predictions (FP) ({len(result["unmatched_predictions"])}):')
            for pred in result['unmatched_predictions'][:10]:  # Show first 10
                pred_text = pred["text"]
                pred_start = pred["start"]
                pred_end = pred["end"]
                pred_label = pred["mapped_label"]
                print(f'  {pred_text} ({pred_start}-{pred_end}) [{pred_label}]')
            
            # Check unmatched ground truth (false negatives)
            print(f'\nUnmatched ground truth (FN) ({len(result["unmatched_ground_truth"])}):')
            for gt in result['unmatched_ground_truth'][:10]:  # Show first 10
                gt_text = gt["span_text"]
                gt_start = gt["start_offset"]
                gt_end = gt["end_offset"]
                gt_type = gt["entity_type"]
                print(f'  {gt_text} ({gt_start}-{gt_end}) [{gt_type}]')
            
            break
    else:
        print('Document not found')

if __name__ == '__main__':
    debug_full_evaluation()