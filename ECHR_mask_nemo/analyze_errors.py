#!/usr/bin/env python3
"""
Analyze false positives and false negatives from NEMO evaluation results
"""

import json
from collections import defaultdict

def analyze_errors():
    # Load evaluation results
    with open('nemo_fixed_evaluation.json', 'r') as f:
        eval_data = json.load(f)
    
    false_positives = defaultdict(list)
    false_negatives = defaultdict(list)
    true_positives = defaultdict(list)
    
    print("🔍 NEMO Error Analysis")
    print("=" * 50)
    
    # Analyze document results
    for doc_result in eval_data['document_results']:
        doc_id = doc_result['doc_id']
        
        # Get the actual document result data
        result = doc_result.get('result', doc_result)
        pred_entities = result.get('pred_entities', [])
        gt_entities = result.get('gt_entities', [])
        matches = result.get('matches', [])
        
        # Get matched entity texts for comparison
        matched_pred_texts = set()
        matched_gt_texts = set()
        
        for match in matches:
            pred_entity = match['pred']
            gt_entity = match['gt']
            matched_pred_texts.add(f"{pred_entity['start']}-{pred_entity['end']}-{pred_entity['text']}")
            matched_gt_texts.add(f"{gt_entity['start']}-{gt_entity['end']}-{gt_entity['text']}")
            
            # Store true positive example
            true_positives[pred_entity['type']].append({
                'doc_id': doc_id,
                'predicted': pred_entity['text'][:50],
                'ground_truth': gt_entity['text'][:50],
                'overlap': match.get('overlap', 1.0)
            })
        
        # Identify false positives (predicted but not matched)
        for pred_entity in pred_entities:
            pred_key = f"{pred_entity['start']}-{pred_entity['end']}-{pred_entity['text']}"
            if pred_key not in matched_pred_texts:
                false_positives[pred_entity['type']].append({
                    'doc_id': doc_id,
                    'text': pred_entity['text'][:100],
                    'start': pred_entity['start'],
                    'end': pred_entity['end']
                })
        
        # Identify false negatives (ground truth but not matched)
        for gt_entity in gt_entities:
            gt_key = f"{gt_entity['start']}-{gt_entity['end']}-{gt_entity['text']}"
            if gt_key not in matched_gt_texts:
                false_negatives[gt_entity['type']].append({
                    'doc_id': doc_id,
                    'text': gt_entity['text'][:100],
                    'start': gt_entity['start'],
                    'end': gt_entity['end']
                })
    
    # Print analysis results
    for entity_type in ['PERSON', 'LOC', 'DATETIME']:
        if entity_type in eval_data['type_metrics']:
            metrics = eval_data['type_metrics'][entity_type]
            
            print(f"\n📊 {entity_type} ANALYSIS")
            print("-" * 40)
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1-Score: {metrics['f1']:.3f}")
            print(f"TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
            
            # Show true positive examples
            if entity_type in true_positives and true_positives[entity_type]:
                print(f"\n✅ TRUE POSITIVE Examples ({entity_type}):")
                for i, tp in enumerate(true_positives[entity_type][:3]):
                    print(f"   {i+1}. Doc {tp['doc_id']}")
                    print(f"      Predicted: \"{tp['predicted']}\"")
                    print(f"      Ground Truth: \"{tp['ground_truth']}\"")
                    print(f"      Overlap: {tp['overlap']:.2f}")
                    print()
            
            # Show false positive examples
            if entity_type in false_positives and false_positives[entity_type]:
                print(f"❌ FALSE POSITIVE Examples ({entity_type}):")
                for i, fp in enumerate(false_positives[entity_type][:5]):
                    print(f"   {i+1}. Doc {fp['doc_id']}: \"{fp['text']}\"")
                    print(f"      Position: {fp['start']}-{fp['end']}")
                print()
            
            # Show false negative examples
            if entity_type in false_negatives and false_negatives[entity_type]:
                print(f"⚠️  FALSE NEGATIVE Examples ({entity_type}):")
                for i, fn in enumerate(false_negatives[entity_type][:5]):
                    print(f"   {i+1}. Doc {fn['doc_id']}: \"{fn['text']}\"")
                    print(f"      Position: {fn['start']}-{fn['end']}")
                print()
    
    # Save detailed examples for presentation
    examples_data = {
        'true_positives': dict(true_positives),
        'false_positives': dict(false_positives),
        'false_negatives': dict(false_negatives),
        'summary': {
            'total_fps': sum(len(fps) for fps in false_positives.values()),
            'total_fns': sum(len(fns) for fns in false_negatives.values()),
            'total_tps': sum(len(tps) for tps in true_positives.values())
        }
    }
    
    with open('error_analysis.json', 'w') as f:
        json.dump(examples_data, f, indent=2)
    
    print("💾 Detailed examples saved to error_analysis.json")
    print(f"\n📈 SUMMARY:")
    print(f"   Total True Positives: {examples_data['summary']['total_tps']:,}")
    print(f"   Total False Positives: {examples_data['summary']['total_fps']:,}")
    print(f"   Total False Negatives: {examples_data['summary']['total_fns']:,}")

if __name__ == "__main__":
    analyze_errors()