#!/usr/bin/env python3
"""
Debug script to analyze the specific case of "Les Salles Sur Verdon, France"
"""

import json
import sys
import os

# Add the current directory to path to import the evaluation functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai4privacy_evaluate import clean_entity_text, merge_consecutive_entities, PII_TO_GT

def debug_specific_case():
    # Load the AI4Privacy output
    with open('output_ai4privacy.json', 'r') as f:
        ai4privacy_data = json.load(f)
    
    target_text = 'Les Salles Sur Verdon, France'
    print(f'Debugging case: "{target_text}"')
    
    # Find the document
    for doc in ai4privacy_data:
        if target_text in doc['original_text']:
            print(f'\nFound in document {doc["id"]}')
            
            # Get the target span
            text_start = doc['original_text'].find(target_text)
            text_end = text_start + len(target_text)
            print(f'Target span: {text_start}-{text_end}')
            
            # Find overlapping AI4Privacy entities
            overlapping_entities = []
            for entity in doc['ai4privacy_detected_pii']:
                if (entity['start'] < text_end and entity['end'] > text_start):
                    overlapping_entities.append(entity)
            
            print(f'\nOverlapping AI4Privacy entities ({len(overlapping_entities)}):')
            for entity in overlapping_entities:
                print(f'  {entity}')
            
            # Step 1: Apply label mapping
            print('\nStep 1: Apply label mapping')
            mapped_entities = []
            for entity in overlapping_entities:
                original_label = entity['label']
                mapped_label = PII_TO_GT.get(original_label)
                print(f'  {original_label} -> {mapped_label}')
                
                if mapped_label is not None:
                    mapped_entity = entity.copy()
                    mapped_entity['text'] = clean_entity_text(entity['text'])
                    mapped_entity['mapped_label'] = mapped_label
                    mapped_entity['original_label'] = original_label
                    mapped_entities.append(mapped_entity)
                    print(f'    Mapped: {mapped_entity}')
            
            # Step 2: Check if entities are consecutive
            print('\nStep 2: Check consecutive entities')
            sorted_entities = sorted(mapped_entities, key=lambda x: x['start'])
            print('Sorted entities:')
            for i, entity in enumerate(sorted_entities):
                print(f'  {i}: {entity["start"]}-{entity["end"]} "{entity["text"]}" ({entity["mapped_label"]})')
                if i > 0:
                    gap = entity['start'] - sorted_entities[i-1]['end']
                    print(f'      Gap from previous: {gap}')
            
            # Step 3: Test merging
            print('\nStep 3: Test merging')
            merged_entities = merge_consecutive_entities(mapped_entities, gap_threshold=5)
            print(f'After merging ({len(merged_entities)} entities):')
            for entity in merged_entities:
                print(f'  {entity}')
            
            # Check ground truth
            print('\nGround truth entities in this area:')
            for entity in doc['ground_truth_entities']:
                if (entity['start_offset'] < text_end and entity['end_offset'] > text_start):
                    print(f'  {entity}')
            
            break
    else:
        print('Text not found in any document')

if __name__ == '__main__':
    debug_specific_case()