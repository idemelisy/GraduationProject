#!/usr/bin/env python3
"""
Extract concrete examples of errors for presentation
"""

import json
import re
from collections import defaultdict

def load_data():
    # Load predictions
    with open('output_nemo_fixed.json', 'r') as f:
        predictions = json.load(f)
    
    # Load ground truth  
    with open('../ECHR_mask/annotations.json', 'r') as f:
        ground_truth = json.load(f)
    
    return predictions, ground_truth

def normalize_entity_type(entity_type):
    """Normalize entity types"""
    type_mapping = {
        'PERSON': 'PERSON',
        'ADDRESS': 'LOC',
        'LOCATION': 'LOC',
        'LOC': 'LOC',
        'DATE_TIME': 'DATETIME',
        'DATETIME': 'DATETIME'
    }
    return type_mapping.get(entity_type.upper(), None)

def entities_overlap(pred, gt, threshold=0.5):
    """Check if entities overlap"""
    pred_start, pred_end = pred['start'], pred['end']
    gt_start, gt_end = gt['start'], gt['end']
    
    overlap_start = max(pred_start, gt_start)
    overlap_end = min(pred_end, gt_end)
    
    if overlap_start >= overlap_end:
        return False
        
    overlap_length = overlap_end - overlap_start
    pred_length = pred_end - pred_start
    gt_length = gt_end - gt_start
    
    min_length = min(pred_length, gt_length)
    return overlap_length / min_length >= threshold if min_length > 0 else False

def extract_examples():
    predictions, ground_truth = load_data()
    
    examples = {
        'true_positives': defaultdict(list),
        'false_positives': defaultdict(list), 
        'false_negatives': defaultdict(list)
    }
    
    print("🔍 Extracting Error Examples")
    print("=" * 50)
    
    # Process first 10 documents for manageable examples
    for doc_idx in range(min(10, len(predictions), len(ground_truth))):
        pred_doc = predictions[doc_idx]
        gt_doc = ground_truth[doc_idx]
        
        # Normalize predictions
        pred_entities = []
        for entity in pred_doc.get('nemo_detected_pii', []):
            entity_type = normalize_entity_type(entity.get('label', ''))
            if entity_type:
                pred_entities.append({
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0),
                    'type': entity_type,
                    'text': entity.get('text', '')
                })
        
        # Normalize ground truth
        gt_entities = []
        for entity in gt_doc.get('annotations', []):
            entity_type = normalize_entity_type(entity.get('entity_type', ''))
            if entity_type:
                gt_entities.append({
                    'start': entity.get('start_offset', 0),
                    'end': entity.get('end_offset', 0),
                    'type': entity_type,
                    'text': entity.get('span_text', '')
                })
        
        # Find matches
        matched_pred = set()
        matched_gt = set()
        
        for i, pred in enumerate(pred_entities):
            for j, gt in enumerate(gt_entities):
                if i in matched_pred or j in matched_gt:
                    continue
                    
                if pred['type'] == gt['type'] and entities_overlap(pred, gt):
                    # True positive
                    examples['true_positives'][pred['type']].append({
                        'doc_id': f'doc_{doc_idx}',
                        'predicted': pred['text'][:60],
                        'ground_truth': gt['text'][:60],
                        'pred_pos': f"{pred['start']}-{pred['end']}",
                        'gt_pos': f"{gt['start']}-{gt['end']}"
                    })
                    matched_pred.add(i)
                    matched_gt.add(j)
                    break
        
        # False positives (predicted but not matched)
        for i, pred in enumerate(pred_entities):
            if i not in matched_pred:
                examples['false_positives'][pred['type']].append({
                    'doc_id': f'doc_{doc_idx}',
                    'text': pred['text'][:80],
                    'position': f"{pred['start']}-{pred['end']}"
                })
        
        # False negatives (ground truth but not matched)
        for j, gt in enumerate(gt_entities):
            if j not in matched_gt:
                examples['false_negatives'][gt['type']].append({
                    'doc_id': f'doc_{doc_idx}',
                    'text': gt['text'][:80],
                    'position': f"{gt['start']}-{gt['end']}"
                })
    
    # Print examples
    for entity_type in ['PERSON', 'LOC', 'DATETIME']:
        print(f"\n📊 {entity_type} EXAMPLES")
        print("-" * 40)
        
        # True Positives
        if examples['true_positives'][entity_type]:
            print(f"✅ TRUE POSITIVES ({entity_type}):")
            for i, tp in enumerate(examples['true_positives'][entity_type][:3]):
                print(f"   {i+1}. {tp['doc_id']}")
                print(f"      Predicted: \"{tp['predicted']}\" ({tp['pred_pos']})")
                print(f"      Truth:     \"{tp['ground_truth']}\" ({tp['gt_pos']})")
            print()
        
        # False Positives  
        if examples['false_positives'][entity_type]:
            print(f"❌ FALSE POSITIVES ({entity_type}):")
            for i, fp in enumerate(examples['false_positives'][entity_type][:5]):
                print(f"   {i+1}. {fp['doc_id']}: \"{fp['text']}\" ({fp['position']})")
            print()
        
        # False Negatives
        if examples['false_negatives'][entity_type]:
            print(f"⚠️  FALSE NEGATIVES ({entity_type}):")
            for i, fn in enumerate(examples['false_negatives'][entity_type][:5]):
                print(f"   {i+1}. {fn['doc_id']}: \"{fn['text']}\" ({fn['position']})")
            print()
    
    # Save examples
    with open('presentation_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    print("💾 Examples saved to presentation_examples.json")
    
    # Summary
    total_tp = sum(len(examples['true_positives'][t]) for t in examples['true_positives'])
    total_fp = sum(len(examples['false_positives'][t]) for t in examples['false_positives']) 
    total_fn = sum(len(examples['false_negatives'][t]) for t in examples['false_negatives'])
    
    print(f"\n📈 SAMPLE SUMMARY (first 10 docs):")
    print(f"   True Positives: {total_tp}")
    print(f"   False Positives: {total_fp}")
    print(f"   False Negatives: {total_fn}")

if __name__ == "__main__":
    extract_examples()