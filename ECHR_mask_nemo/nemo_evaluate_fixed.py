#!/usr/bin/env python3
"""
nemo_evaluate_fixed.py

Evaluate the fixed NEMO output against ground truth annotations.
"""

import json
import re
from collections import defaultdict

def load_ground_truth(gt_path):
    """Load ground truth annotations"""
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_annotations = {}
    for doc_data in data:
        doc_id = doc_data.get('doc_id', 'unknown')
        annotations = doc_data.get('annotations', [])
        standardized = []
        for ann in annotations:
            standardized.append({
                'start': ann.get('start_offset', ann.get('start', 0)),
                'end': ann.get('end_offset', ann.get('end', 0)),
                'label': ann.get('entity_type', ann.get('type', ann.get('label', ''))),
                'text': ann.get('span_text', ann.get('text', ''))
            })
        
        gt_annotations[doc_id] = standardized
    
    return gt_annotations

def load_nemo_predictions(nemo_path):
    """Load NEMO predictions"""
    with open(nemo_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = {}
    for item in data:
        doc_id = item.get('id', f'doc_{len(predictions)}')
        entities = item.get('nemo_detected_pii', [])
        # Standardize entity format
        standardized_entities = []
        for entity in entities:
            standardized_entities.append({
                'start': entity.get('start', entity.get('start_offset', 0)),
                'end': entity.get('end', entity.get('end_offset', 0)),
                'label': entity.get('label', entity.get('type', '')),
                'text': entity.get('text', '')
            })
        predictions[doc_id] = standardized_entities
    
    return predictions

def normalize_entity_type(entity_type):
    """Normalize entity types for comparison - focus only on key types"""
    # Map NEMO types to ground truth types for key entities only
    type_mapping = {
        'PERSON': 'PERSON',
        'ADDRESS': 'LOC',  # Address maps to location
        'LOCATION': 'LOC',
        'LOC': 'LOC',      # Ground truth LOC maps to LOC
        'DATE_TIME': 'DATETIME',
        'DATETIME': 'DATETIME'
    }
    
    normalized = type_mapping.get(entity_type.upper(), None)
    return normalized  # Return None for types we don't care about

def calculate_overlap(pred_start, pred_end, gt_start, gt_end):
    """Calculate overlap ratio between predicted and ground truth spans"""
    overlap_start = max(pred_start, gt_start)
    overlap_end = min(pred_end, gt_end)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_length = overlap_end - overlap_start
    pred_length = pred_end - pred_start
    gt_length = gt_end - gt_start
    
    # Calculate overlap as percentage of the smaller span
    min_length = min(pred_length, gt_length)
    return overlap_length / min_length if min_length > 0 else 0.0

def evaluate_document(pred_entities, gt_entities, overlap_threshold=0.5):
    """Evaluate predictions for a single document"""
    
    # Normalize entity types and field names - filter for key types only
    pred_normalized = []
    for entity in pred_entities:
        entity_type = normalize_entity_type(entity.get('type', entity.get('label', '')))
        if entity_type:  # Only include entities we care about
            pred_normalized.append({
                'start': entity.get('start', entity.get('start_offset', 0)),
                'end': entity.get('end', entity.get('end_offset', 0)),
                'type': entity_type,
                'text': entity.get('text', '')
            })
    
    gt_normalized = []
    for entity in gt_entities:
        entity_type = normalize_entity_type(entity.get('type', entity.get('label', '')))
        if entity_type:  # Only include entities we care about
            gt_normalized.append({
                'start': entity.get('start', entity.get('start_offset', 0)),
                'end': entity.get('end', entity.get('end_offset', 0)),
                'type': entity_type,
                'text': entity.get('text', '')
            })
    
    # Track matches
    pred_matched = [False] * len(pred_normalized)
    gt_matched = [False] * len(gt_normalized)
    
    matches = []
    
    # Find matches
    for i, pred in enumerate(pred_normalized):
        best_match = None
        best_overlap = 0.0
        best_j = -1
        
        for j, gt in enumerate(gt_normalized):
            if gt_matched[j]:
                continue
                
            # Check if types match
            if pred['type'] != gt['type']:
                continue
            
            # Calculate overlap
            overlap = calculate_overlap(pred['start'], pred['end'], gt['start'], gt['end'])
            
            if overlap >= overlap_threshold and overlap > best_overlap:
                best_overlap = overlap
                best_match = gt
                best_j = j
        
        if best_match:
            pred_matched[i] = True
            gt_matched[best_j] = True
            matches.append({
                'pred': pred,
                'gt': best_match,
                'overlap': best_overlap
            })
    
    # Calculate metrics
    true_positives = len(matches)
    false_positives = len(pred_normalized) - true_positives
    false_negatives = len(gt_normalized) - true_positives
    
    precision = true_positives / len(pred_normalized) if len(pred_normalized) > 0 else 0.0
    recall = true_positives / len(gt_normalized) if len(gt_normalized) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'matches': matches,
        'pred_entities': pred_normalized,
        'gt_entities': gt_normalized
    }

def evaluate_by_entity_type(predictions, ground_truth):
    """Evaluate performance by entity type"""
    
    type_metrics = defaultdict(lambda: {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    })
    
    overall_metrics = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    doc_results = []
    
    # Convert to lists for index-based matching
    gt_docs = list(ground_truth.values())
    pred_docs = list(predictions.values())
    gt_ids = list(ground_truth.keys())
    pred_ids = list(predictions.keys())
    
    # Match by index since IDs don't align
    for i in range(min(len(gt_docs), len(pred_docs))):
        gt_entities = gt_docs[i]
        pred_entities = pred_docs[i]
        doc_id = f"doc_{i}"
        
        doc_result = evaluate_document(pred_entities, gt_entities)
        doc_results.append({
            'doc_id': doc_id,
            'gt_id': gt_ids[i] if i < len(gt_ids) else f"gt_{i}",
            'pred_id': pred_ids[i] if i < len(pred_ids) else f"pred_{i}",
            'result': doc_result
        })
        
        # Accumulate overall metrics
        overall_metrics['true_positives'] += doc_result['true_positives']
        overall_metrics['false_positives'] += doc_result['false_positives']
        overall_metrics['false_negatives'] += doc_result['false_negatives']
        
        # Accumulate type-specific metrics - only for entities we care about
        for entity in doc_result['pred_entities']:
            entity_type = entity['type']
            if entity_type:  # Only process entities we care about
                # Check if this entity was matched
                matched = any(m['pred'] == entity for m in doc_result['matches'])
                if matched:
                    type_metrics[entity_type]['true_positives'] += 1
                else:
                    type_metrics[entity_type]['false_positives'] += 1
        
        for entity in doc_result['gt_entities']:
            entity_type = entity['type']
            if entity_type:  # Only process entities we care about
                # Check if this entity was matched
                matched = any(m['gt'] == entity for m in doc_result['matches'])
                if not matched:
                    type_metrics[entity_type]['false_negatives'] += 1
    
    # Calculate overall metrics
    tp = overall_metrics['true_positives']
    fp = overall_metrics['false_positives']
    fn = overall_metrics['false_negatives']
    
    overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    overall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Calculate type-specific metrics
    for entity_type in type_metrics:
        tp = type_metrics[entity_type]['true_positives']
        fp = type_metrics[entity_type]['false_positives']
        fn = type_metrics[entity_type]['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        type_metrics[entity_type]['precision'] = precision
        type_metrics[entity_type]['recall'] = recall
        type_metrics[entity_type]['f1'] = f1
    
    return {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        },
        'by_type': dict(type_metrics),
        'doc_results': doc_results
    }

def main():
    # File paths
    annotations_path = "/home/ide/ide/ECHR_mask/annotations.json"
    nemo_path = "/home/ide/ide/ECHR_mask_nemo/output_nemo_fixed.json"
    output_path = "/home/ide/ide/ECHR_mask_nemo/nemo_fixed_evaluation.json"
    
    print("🔍 NEMO Fixed Output Evaluation")
    print("=" * 50)
    
    # Load data
    print("📂 Loading data...")
    ground_truth = load_ground_truth(annotations_path)
    predictions = load_nemo_predictions(nemo_path)
    
    print(f"   Ground truth documents: {len(ground_truth)}")
    print(f"   Prediction documents: {len(predictions)}")
    
    # Count total entities
    total_gt_entities = sum(len(entities) for entities in ground_truth.values())
    total_pred_entities = sum(len(entities) for entities in predictions.values())
    
    print(f"   Ground truth entities: {total_gt_entities}")
    print(f"   Predicted entities: {total_pred_entities}")
    
    # Run evaluation
    print("\n🧮 Running evaluation...")
    results = evaluate_by_entity_type(predictions, ground_truth)
    
    # Print results
    print("\n📊 Overall Results:")
    print("-" * 30)
    overall = results['overall']
    print(f"Precision: {overall['precision']:.3f}")
    print(f"Recall:    {overall['recall']:.3f}")
    print(f"F1-Score:  {overall['f1']:.3f}")
    print(f"True Positives:  {overall['true_positives']}")
    print(f"False Positives: {overall['false_positives']}")
    print(f"False Negatives: {overall['false_negatives']}")
    
    print("\n📋 Results by Entity Type:")
    print("-" * 50)
    print(f"{'Type':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 50)
    
    for entity_type, metrics in sorted(results['by_type'].items()):
        print(f"{entity_type:<10} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
              f"{metrics['f1']:<10.3f} {metrics['true_positives']:<6} "
              f"{metrics['false_positives']:<6} {metrics['false_negatives']:<6}")
    
    # Find best and worst performing documents
    doc_f1_scores = [(dr['doc_id'], dr['result']['f1']) for dr in results['doc_results']]
    doc_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Best performing documents:")
    for doc_id, f1 in doc_f1_scores[:5]:
        print(f"   {doc_id}: F1 = {f1:.3f}")
    
    print(f"\n⚠️  Worst performing documents:")
    for doc_id, f1 in doc_f1_scores[-5:]:
        print(f"   {doc_id}: F1 = {f1:.3f}")
    
    # Show some example matches and misses
    print(f"\n🎯 Example Analysis (from {doc_f1_scores[0][0]}):")
    best_doc_result = next(dr['result'] for dr in results['doc_results'] if dr['doc_id'] == doc_f1_scores[0][0])
    
    if best_doc_result['matches']:
        print("✅ Example matches:")
        for match in best_doc_result['matches'][:3]:
            print(f"   Predicted: {match['pred']['type']} '{match['pred']['text'][:30]}...'")
            print(f"   Ground Truth: {match['gt']['type']} '{match['gt']['text'][:30]}...'")
            print(f"   Overlap: {match['overlap']:.2f}")
            print()
    
    # Save detailed results
    print(f"\n💾 Saving detailed results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_metrics': results['overall'],
            'type_metrics': results['by_type'],
            'document_results': [
                {
                    'doc_id': dr['doc_id'],
                    'precision': dr['result']['precision'],
                    'recall': dr['result']['recall'],
                    'f1': dr['result']['f1'],
                    'true_positives': dr['result']['true_positives'],
                    'false_positives': dr['result']['false_positives'],
                    'false_negatives': dr['result']['false_negatives']
                }
                for dr in results['doc_results']
            ],
            'evaluation_settings': {
                'overlap_threshold': 0.5,
                'entity_type_mapping': {
                    'PERSON': 'PER',
                    'DATE_TIME': 'MISC',
                    'ADDRESS': 'LOC'
                }
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📈 Summary: Fixed NEMO achieved {overall['f1']:.3f} F1-score")

if __name__ == "__main__":
    main()