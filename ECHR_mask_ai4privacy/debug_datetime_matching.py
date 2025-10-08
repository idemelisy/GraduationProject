#!/usr/bin/env python3
"""
Debug script to test why specific DATETIME entities aren't being matched
"""

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai4privacy_evaluate import evaluate_single_document, PII_TO_GT, clean_entity_text, merge_consecutive_entities

def debug_datetime_matching():
    # Load AI4Privacy output
    with open('output_ai4privacy.json', 'r') as f:
        ai4privacy_data = json.load(f)
    
    # Find document 001-84741
    for doc in ai4privacy_data:
        if doc['id'] == '001-84741':
            print(f'Debugging document {doc["id"]}')
            
            # Get predicted entities around the problematic positions
            target_positions = [(1313, 1328), (1345, 1357)]
            
            for target_start, target_end in target_positions:
                print(f'\n=== Debugging position {target_start}-{target_end} ===')
                
                # Find AI4Privacy entities near this position
                nearby_preds = []
                for entity in doc['ai4privacy_detected_pii']:
                    if (entity['start'] >= target_start - 20 and 
                        entity['start'] <= target_end + 20):
                        nearby_preds.append(entity)
                
                print(f'AI4Privacy entities near position:')
                for entity in nearby_preds:
                    print(f'  {entity}')
                
                # Apply our post-processing
                mapped_preds = []
                for entity in nearby_preds:
                    mapped_label = PII_TO_GT.get(entity['label'])
                    if mapped_label:
                        mapped_entity = entity.copy()
                        mapped_entity['text'] = clean_entity_text(entity['text'])
                        mapped_entity['mapped_label'] = mapped_label
                        mapped_entity['original_label'] = entity['label']
                        mapped_preds.append(mapped_entity)
                
                print(f'\\nAfter label mapping:')
                for entity in mapped_preds:
                    print(f'  {entity}')
                
                # Test merging
                merged_preds = merge_consecutive_entities(mapped_preds)
                print(f'\\nAfter merging:')
                for entity in merged_preds:
                    print(f'  {entity}')
                
                # Find corresponding ground truth
                gt_entities = []
                for gt in doc['ground_truth_entities']:
                    if (gt['start_offset'] >= target_start - 20 and 
                        gt['start_offset'] <= target_end + 20 and
                        gt.get('annotator') == 'annotator1'):
                        gt_entities.append(gt)
                
                print(f'\\nGround truth entities:')
                for gt in gt_entities:
                    print(f'  {gt}')
                
                # Test the evaluation on just these entities
                if merged_preds and gt_entities:
                    print(f'\\nTesting evaluation...')
                    result = evaluate_single_document(merged_preds, gt_entities)
                    print(f'Result: {result["tp_relaxed"]} relaxed TP, {result["fp"]} FP, {result["fn"]} FN')
                
            break
    else:
        print('Document not found')

if __name__ == '__main__':
    debug_datetime_matching()