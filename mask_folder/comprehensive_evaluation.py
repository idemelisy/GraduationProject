#!/usr/bin/env python3
"""
Comprehensive PII Model Evaluation Script
Adapted for your specific model outputs and evaluation framework
"""

import pandas as pd
import ast
import re
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support

# ----------------------
# 1️⃣ Configuration and Setup
# ----------------------

# Model file configurations
MODEL_CONFIGS = {
    'piranha': {
        'file': 'output_piranha.xlsx',
        'detection_col': 'Detected_PII',
        'entity_key': 'entity_group',
        'text_key': 'word'
    },
    'llama': {
        'file': 'output_llama.xlsx', 
        'detection_col': 'llama_pii',
        'entity_key': 'label',
        'text_key': 'text'
    },
    'ai4privacy': {
        'file': 'output_ai4privacy.xlsx',
        'detection_col': 'ai4privacy_detected_pii',
        'entity_key': None,  # Different format - dict with lists
        'text_key': None
    },
    'presidio': {
        'file': 'output_presidio.xlsx',
        'detection_col': 'presidio_detected_pii', 
        'entity_key': 'label',
        'text_key': 'text'
    },
    'hf': {
        'file': 'output_hf.xlsx',
        'detection_col': 'hf_detected_pii',
        'entity_key': 'label', 
        'text_key': 'text'
    }
}

# Enhanced label mapping based on your models
LABEL_MAPPING = {
    # Piranha mappings
    "GIVENNAME": "PERSON_FIRST",
    "SURNAME": "PERSON_LAST", 
    "FAMILYNAME": "PERSON_LAST",
    "DATEOFBIRTH": "DATE_BIRTH",
    "CITY": "LOCATION_CITY",
    
    # LLaMA mappings
    "GIVENAME": "PERSON_FIRST",
    
    # Presidio mappings  
    "PERSON": "PERSON_FIRST",  # May need refinement
    "DATE_TIME": "DATE_BIRTH", # May catch more than birth dates
    "LOCATION": "LOCATION_CITY",
    "NRP": "NATIONALITY",  # New category
    
    # HF mappings
    "PER": "PERSON_FIRST",
    "LOC": "LOCATION_CITY", 
    "ORG": "ORGANIZATION",  # New category
    "MISC": "MISCELLANEOUS", # New category
    
    # AI4Privacy mappings
    "DATE": "DATE_BIRTH",
    "TITLE": "TITLE",  # New category
    "BUILDINGNUM": "LOCATION_ADDRESS",
    
    # Common variations
    "BIRTHDATE": "DATE_BIRTH",
    "DATE OF BIRTH": "DATE_BIRTH",
    "LOCATION_ADDRESS": "LOCATION_ADDRESS", 
    "FIRSTNAME": "PERSON_FIRST",
    "LASTNAME": "PERSON_LAST",
}

def normalize_label(model_label):
    """Normalize entity labels across different models"""
    if model_label is None:
        return "UNKNOWN"
    return LABEL_MAPPING.get(model_label.upper(), model_label.upper())

# ----------------------
# 2️⃣ Data Loading and Parsing Functions
# ----------------------

def parse_detection_data(data_str, config):
    """Parse detection data based on model format"""
    if pd.isna(data_str) or not data_str:
        return []
    
    try:
        if isinstance(data_str, str):
            parsed = ast.literal_eval(data_str)
        else:
            parsed = data_str
            
        entities = []
        
        if config['entity_key'] is None:  # AI4Privacy format - dict with lists
            if isinstance(parsed, dict):
                for entity_type, entity_list in parsed.items():
                    if isinstance(entity_list, list):
                        for entity_text in entity_list:
                            entities.append({
                                'label': normalize_label(entity_type),
                                'text': entity_text.strip()
                            })
        else:  # Standard format - list of dicts
            if isinstance(parsed, list):
                for entity in parsed:
                    if isinstance(entity, dict):
                        label = entity.get(config['entity_key'], 'UNKNOWN')
                        text = entity.get(config['text_key'], '')
                        entities.append({
                            'label': normalize_label(label),
                            'text': str(text).strip()
                        })
        
        return entities
        
    except Exception as e:
        print(f"Error parsing detection data: {e}")
        return []

def extract_ground_truth_entities(original_text, masked_text):
    """Extract ground truth entities by comparing original and masked text"""
    if pd.isna(original_text) or pd.isna(masked_text):
        return []
    
    entities = []
    
    # Find all mask tokens in masked text
    mask_pattern = r'\[([^\]]+)\]'
    mask_matches = list(re.finditer(mask_pattern, masked_text))
    
    if not mask_matches:
        return entities
    
    # Process mask matches to extract original entities
    current_pos = 0
    orig_pos = 0
    
    for match in mask_matches:
        entity_type = match.group(1)
        mask_start = match.start()
        mask_end = match.end()
        
        # Move original position to match the text before this mask
        text_before_mask = masked_text[current_pos:mask_start]
        orig_pos += len(text_before_mask)
        
        # Find the next matching point after the mask
        text_after_mask = masked_text[mask_end:mask_end+50]  # Look ahead 50 chars
        
        # Clean the text after mask for better matching
        words_after = text_after_mask.split()[:5]  # First 5 words after mask
        if words_after:
            search_text = ' '.join(words_after)
            # Clean up weird characters
            search_text = re.sub(r'â[^\s]*', '', search_text)  # Remove â characters
            search_text = search_text.strip()
            
            if len(search_text) > 3:  # Only search if we have meaningful text
                # Find this text in original starting from our current position
                remaining_original = original_text[orig_pos:]
                next_match = remaining_original.find(search_text)
                
                if next_match >= 0:
                    # Extract the entity text
                    entity_text = remaining_original[:next_match].strip()
                    
                    # Clean up entity text
                    entity_text = re.sub(r'^[^\w]*|[^\w]*$', '', entity_text)  # Remove non-word chars at start/end
                    entity_text = re.sub(r'\s+', ' ', entity_text)  # Normalize spaces
                    
                    if entity_text and len(entity_text) > 0:
                        entities.append({
                            'label': normalize_label(entity_type),
                            'text': entity_text
                        })
                    
                    # Update position
                    orig_pos += next_match + len(search_text)
                else:
                    # Fallback: assume reasonable entity length
                    window = remaining_original[:30]  # Look at next 30 chars
                    words = window.split()
                    if words:
                        # Take 1-2 words as entity
                        entity_text = ' '.join(words[:2]).strip()
                        entity_text = re.sub(r'[^\w\s-]', '', entity_text)  # Keep only words, spaces, hyphens
                        if entity_text:
                            entities.append({
                                'label': normalize_label(entity_type),
                                'text': entity_text
                            })
        
        current_pos = mask_end
    
    return entities

def load_all_model_data():
    """Load and combine data from all model outputs"""
    print("📂 Loading model data...")
    
    # Start with input file for ground truth
    input_df = pd.read_excel('input_10.xlsx')
    
    # Extract ground truth entities
    print("🎯 Extracting ground truth entities...")
    ground_truth_entities = []
    for idx, row in input_df.iterrows():
        gt_entities = extract_ground_truth_entities(row['original_text'], row['masked_text'])
        ground_truth_entities.append(gt_entities)
    
    # Create master dataframe
    master_df = input_df.copy()
    master_df['GT_entities'] = ground_truth_entities
    
    # Load each model's predictions
    for model_name, config in MODEL_CONFIGS.items():
        try:
            print(f"📊 Loading {model_name.upper()} predictions...")
            model_df = pd.read_excel(config['file'])
            
            # Parse detection data
            detection_col = config['detection_col'] 
            parsed_entities = []
            
            for idx, row in model_df.iterrows():
                entities = parse_detection_data(row[detection_col], config)
                parsed_entities.append(entities)
            
            # Add to master dataframe
            master_df[f'{model_name}_detected_pii'] = parsed_entities
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            master_df[f'{model_name}_detected_pii'] = [[] for _ in range(len(master_df))]
    
    return master_df

# ----------------------
# 3️⃣ Evaluation Functions
# ----------------------

def extract_entity_labels(entity_list):
    """Extract normalized labels from entity list"""
    if not isinstance(entity_list, list):
        return []
    return [entity.get('label', 'UNKNOWN') for entity in entity_list if isinstance(entity, dict)]

def extract_entity_texts(entity_list):
    """Extract entity texts from entity list"""
    if not isinstance(entity_list, list):
        return []
    return [entity.get('text', '') for entity in entity_list if isinstance(entity, dict)]

def evaluate_model_overall(model_column, df):
    """Evaluate model performance overall using micro-averaging"""
    y_true = []
    y_pred = []
    
    # Collect all unique labels
    all_labels = set()
    for gt_entities in df['GT_entities']:
        all_labels.update(extract_entity_labels(gt_entities))
    for pred_entities in df[model_column]:
        all_labels.update(extract_entity_labels(pred_entities))
    
    # Create binary classification problem
    for idx, (gt_entities, pred_entities) in enumerate(zip(df['GT_entities'], df[model_column])):
        gt_labels = set(extract_entity_labels(gt_entities))
        pred_labels = set(extract_entity_labels(pred_entities))
        
        # For each unique label, create binary classification
        for label in all_labels:
            y_true.append(1 if label in gt_labels else 0)
            y_pred.append(1 if label in pred_labels else 0)
    
    if not y_true:
        return 0.0, 0.0, 0.0
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    return precision, recall, f1

def evaluate_model_by_entity_type(model_column, df):
    """Evaluate model performance by entity type"""
    # Collect all entity types
    entity_types = set()
    for gt_entities in df['GT_entities']:
        entity_types.update(extract_entity_labels(gt_entities))
    for pred_entities in df[model_column]:
        entity_types.update(extract_entity_labels(pred_entities))
    
    results = {}
    
    for entity_type in entity_types:
        y_true = []
        y_pred = []
        
        for gt_entities, pred_entities in zip(df['GT_entities'], df[model_column]):
            gt_labels = extract_entity_labels(gt_entities)
            pred_labels = extract_entity_labels(pred_entities)
            
            y_true.append(1 if entity_type in gt_labels else 0)
            y_pred.append(1 if entity_type in pred_labels else 0)
        
        if any(y_true) or any(y_pred):  # Only evaluate if entity type appears
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            results[entity_type] = {
                'precision': precision,
                'recall': recall, 
                'f1': f1,
                'gt_count': sum(y_true),
                'pred_count': sum(y_pred)
            }
    
    return results

# ----------------------
# 4️⃣ Main Evaluation and Reporting
# ----------------------

def main():
    print("🚀 COMPREHENSIVE PII MODEL EVALUATION")
    print("=" * 50)
    
    # Load all data
    df = load_all_model_data()
    print(f"✅ Loaded data for {len(df)} documents\n")
    
    # Get model columns
    model_columns = [col for col in df.columns if col.endswith('_detected_pii')]
    
    print(f"🔍 Evaluating {len(model_columns)} models...")
    print("Models:", [col.replace('_detected_pii', '').upper() for col in model_columns])
    print()
    
    # Overall evaluation
    print("🏆 OVERALL PERFORMANCE RESULTS")
    print("{:<15} | {:<9} | {:<9} | {:<9}".format("Model", "Precision", "Recall", "F1"))
    print("-" * 55)
    
    overall_results = {}
    for col in model_columns:
        model_name = col.replace('_detected_pii', '')
        precision, recall, f1 = evaluate_model_overall(col, df)
        overall_results[model_name] = {
            'precision': precision,
            'recall': recall, 
            'f1': f1
        }
        print("{:<15} | {:<9.3f} | {:<9.3f} | {:<9.3f}".format(
            model_name.upper(), precision, recall, f1
        ))
    
    # Rank by F1 score
    ranked_models = sorted(overall_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    print(f"\n🥇 Best overall: {ranked_models[0][0].upper()} (F1={ranked_models[0][1]['f1']:.3f})")
    
    # Per-entity type evaluation
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE BY ENTITY TYPE")
    print("=" * 60)
    
    # Collect all entity types across models
    all_entity_types = set()
    for col in model_columns:
        model_name = col.replace('_detected_pii', '')
        entity_results = evaluate_model_by_entity_type(col, df)
        all_entity_types.update(entity_results.keys())
    
    # Display results by entity type
    for entity_type in sorted(all_entity_types):
        print(f"\n=== {entity_type} ===")
        type_results = []
        
        for col in model_columns:
            model_name = col.replace('_detected_pii', '')
            entity_results = evaluate_model_by_entity_type(col, df)
            
            if entity_type in entity_results:
                result = entity_results[entity_type]
                type_results.append((model_name, result))
        
        # Sort by F1 score
        type_results.sort(key=lambda x: x[1]['f1'], reverse=True)
        
        for i, (model_name, result) in enumerate(type_results, 1):
            if result['f1'] > 0:
                print(f"{i}. {model_name.upper():<12}: F1={result['f1']:.3f} "
                      f"(P={result['precision']:.3f}, R={result['recall']:.3f}) "
                      f"[GT:{result['gt_count']}, Pred:{result['pred_count']}]")
            else:
                print(f"{i}. {model_name.upper():<12}: No detection")
    
    # Summary recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    best_overall = ranked_models[0][0]
    print(f"🎯 Overall best model: {best_overall.upper()}")
    
    # Find best model per entity type
    entity_champions = {}
    for entity_type in all_entity_types:
        best_f1 = 0
        best_model = None
        
        for col in model_columns:
            model_name = col.replace('_detected_pii', '')
            entity_results = evaluate_model_by_entity_type(col, df)
            
            if entity_type in entity_results:
                f1 = entity_results[entity_type]['f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
        
        if best_model and best_f1 > 0:
            entity_champions[entity_type] = (best_model, best_f1)
    
    print("\n🏅 Best models by entity type:")
    for entity_type, (model, f1) in entity_champions.items():
        print(f"  {entity_type:<20}: {model.upper():<10} (F1={f1:.3f})")
    
    print(f"\n📈 Total ground truth entities: {sum([len(entities) for entities in df['GT_entities']])}")
    print(f"📊 Unique entity types found: {len(all_entity_types)}")
    print(f"🔬 Documents evaluated: {len(df)}")

if __name__ == "__main__":
    main()