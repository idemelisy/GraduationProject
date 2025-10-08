#!/usr/bin/env python3
"""
Detailed PII Performance Analysis with Confusion Matrices and Heat Maps
Shows True Positives, False Positives, True Negatives, False Negatives for each model
"""

import pandas as pd
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# Configuration (same as before)
# ----------------------

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
        'entity_key': None,
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

LABEL_MAPPING = {
    "GIVENNAME": "PERSON_FIRST",
    "SURNAME": "PERSON_LAST", 
    "FAMILYNAME": "PERSON_LAST",
    "DATEOFBIRTH": "DATE_BIRTH",
    "CITY": "LOCATION_CITY",
    "GIVENAME": "PERSON_FIRST",
    "PERSON": "PERSON_FIRST",
    "DATE_TIME": "DATE_BIRTH",
    "LOCATION": "LOCATION_CITY",
    "NRP": "NATIONALITY",
    "PER": "PERSON_FIRST",
    "LOC": "LOCATION_CITY", 
    "ORG": "ORGANIZATION",
    "MISC": "MISCELLANEOUS",
    "DATE": "DATE_BIRTH",
    "TITLE": "TITLE",
    "BUILDINGNUM": "LOCATION_ADDRESS",
    "BIRTHDATE": "DATE_BIRTH",
    "DATE OF BIRTH": "DATE_BIRTH",
    "LOCATION_ADDRESS": "LOCATION_ADDRESS", 
    "FIRSTNAME": "PERSON_FIRST",
    "LASTNAME": "PERSON_LAST",
}

def normalize_label(model_label):
    if model_label is None:
        return "UNKNOWN"
    return LABEL_MAPPING.get(model_label.upper(), model_label.upper())

# ----------------------
# Data Loading Functions (simplified versions from before)
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
        
        if config['entity_key'] is None:  # AI4Privacy format
            if isinstance(parsed, dict):
                for entity_type, entity_list in parsed.items():
                    if isinstance(entity_list, list):
                        for entity_text in entity_list:
                            entities.append({
                                'label': normalize_label(entity_type),
                                'text': entity_text.strip()
                            })
        else:  # Standard format
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
        return []

def extract_ground_truth_entities(original_text, masked_text):
    """Simple ground truth extraction"""
    if pd.isna(original_text) or pd.isna(masked_text):
        return []
    
    entities = []
    mask_pattern = r'\[([^\]]+)\]'
    mask_matches = list(re.finditer(mask_pattern, masked_text))
    
    for match in mask_matches:
        entity_type = match.group(1)
        entities.append({
            'label': normalize_label(entity_type),
            'text': f"ENTITY_{normalize_label(entity_type)}"  # Simplified for evaluation
        })
    
    return entities

# ----------------------
# Detailed Performance Analysis Functions
# ----------------------

def calculate_confusion_matrix_data(gt_entities_list, pred_entities_list, entity_types):
    """Calculate TP, FP, TN, FN for each entity type across all documents"""
    results = {}
    
    for entity_type in entity_types:
        tp = 0  # True Positives
        fp = 0  # False Positives  
        tn = 0  # True Negatives
        fn = 0  # False Negatives
        
        for gt_entities, pred_entities in zip(gt_entities_list, pred_entities_list):
            # Extract labels for this document
            gt_labels = set([e['label'] for e in gt_entities if isinstance(e, dict)])
            pred_labels = set([e['label'] for e in pred_entities if isinstance(e, dict)])
            
            # Calculate metrics for this entity type
            gt_has_entity = entity_type in gt_labels
            pred_has_entity = entity_type in pred_labels
            
            if gt_has_entity and pred_has_entity:
                tp += 1
            elif not gt_has_entity and pred_has_entity:
                fp += 1
            elif gt_has_entity and not pred_has_entity:
                fn += 1
            else:  # not gt_has_entity and not pred_has_entity
                tn += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        results[entity_type] = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy
        }
    
    return results

def create_performance_heatmap(model_results, entity_types, metric='f1'):
    """Create heatmap showing model performance across entity types"""
    
    # Create matrix
    models = list(model_results.keys())
    matrix = np.zeros((len(models), len(entity_types)))
    
    for i, model in enumerate(models):
        for j, entity_type in enumerate(entity_types):
            if entity_type in model_results[model]:
                matrix[i, j] = model_results[model][entity_type][metric]
            else:
                matrix[i, j] = 0.0
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, 
                xticklabels=entity_types, 
                yticklabels=[m.upper() for m in models],
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                vmin=0, 
                vmax=1,
                cbar_kws={'label': metric.upper()})
    
    plt.title(f'PII Detection Performance Heatmap ({metric.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('PII Entity Types', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'pii_performance_heatmap_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return matrix

def print_detailed_confusion_matrices(model_results):
    """Print detailed confusion matrices for each model"""
    
    print("📊 DETAILED CONFUSION MATRIX ANALYSIS")
    print("=" * 80)
    
    for model_name, entity_results in model_results.items():
        print(f"\n🔍 {model_name.upper()} DETAILED PERFORMANCE:")
        print("-" * 50)
        
        # Calculate totals
        total_tp = sum([r['tp'] for r in entity_results.values()])
        total_fp = sum([r['fp'] for r in entity_results.values()])  
        total_tn = sum([r['tn'] for r in entity_results.values()])
        total_fn = sum([r['fn'] for r in entity_results.values()])
        
        print(f"OVERALL: TP={total_tp}, FP={total_fp}, TN={total_tn}, FN={total_fn}")
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        print(f"METRICS: P={overall_precision:.3f}, R={overall_recall:.3f}, F1={overall_f1:.3f}")
        print()
        
        # Per-entity breakdown
        print("PER ENTITY TYPE:")
        print(f"{'Entity':<18} | {'TP':<3} | {'FP':<3} | {'TN':<3} | {'FN':<3} | {'Prec':<5} | {'Rec':<5} | {'F1':<5} | {'Acc':<5}")
        print("-" * 85)
        
        for entity_type in sorted(entity_results.keys()):
            r = entity_results[entity_type]
            if r['tp'] > 0 or r['fp'] > 0 or r['fn'] > 0:  # Only show entities that were detected or missed
                print(f"{entity_type:<18} | {r['tp']:<3} | {r['fp']:<3} | {r['tn']:<3} | {r['fn']:<3} | "
                      f"{r['precision']:<5.3f} | {r['recall']:<5.3f} | {r['f1']:<5.3f} | {r['accuracy']:<5.3f}")

def create_model_comparison_charts(model_results, entity_types):
    """Create comparison charts for different metrics"""
    
    metrics = ['precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create data for this metric
        model_names = list(model_results.keys())
        metric_data = []
        
        for model in model_names:
            # Calculate average metric across entity types
            values = [model_results[model][et][metric] for et in entity_types 
                     if et in model_results[model]]
            avg_metric = np.mean(values) if values else 0.0
            metric_data.append(avg_metric)
        
        # Bar chart
        bars = ax.bar([m.upper() for m in model_names], metric_data, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax.set_title(f'Average {metric.upper()} Score', fontweight='bold')
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Fourth subplot: Entity type coverage
    ax = axes[3]
    coverage_data = []
    for model in model_names:
        # Count how many entity types this model can detect
        detected_types = sum(1 for et in entity_types 
                           if et in model_results[model] and model_results[model][et]['tp'] > 0)
        coverage_data.append(detected_types)
    
    bars = ax.bar([m.upper() for m in model_names], coverage_data,
                 color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax.set_title('Entity Type Coverage', fontweight='bold')
    ax.set_ylabel('Number of Entity Types Detected')
    ax.set_ylim(0, len(entity_types))
    
    for bar, value in zip(bars, coverage_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------
# Main Analysis Function
# ----------------------

def main():
    print("🚀 DETAILED PII PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("📂 Loading data...")
    input_df = pd.read_excel('input_10.xlsx')
    
    # Extract ground truth
    ground_truth_entities = []
    for idx, row in input_df.iterrows():
        gt_entities = extract_ground_truth_entities(row['original_text'], row['masked_text'])
        ground_truth_entities.append(gt_entities)
    
    # Load model predictions
    all_model_results = {}
    model_predictions = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        try:
            print(f"📊 Loading {model_name.upper()} predictions...")
            model_df = pd.read_excel(config['file'])
            
            predictions = []
            for idx, row in model_df.iterrows():
                entities = parse_detection_data(row[config['detection_col']], config)
                predictions.append(entities)
            
            model_predictions[model_name] = predictions
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            model_predictions[model_name] = [[] for _ in range(len(input_df))]
    
    # Get all entity types
    all_entity_types = set()
    for gt_list in ground_truth_entities:
        for entity in gt_list:
            if isinstance(entity, dict):
                all_entity_types.add(entity['label'])
    
    for pred_list in model_predictions.values():
        for doc_preds in pred_list:
            for entity in doc_preds:
                if isinstance(entity, dict):
                    all_entity_types.add(entity['label'])
    
    all_entity_types = sorted(list(all_entity_types))
    print(f"✅ Found {len(all_entity_types)} entity types: {', '.join(all_entity_types)}")
    
    # Calculate detailed performance for each model
    print("\n🔍 Calculating detailed performance metrics...")
    
    for model_name, predictions in model_predictions.items():
        print(f"   Analyzing {model_name.upper()}...")
        results = calculate_confusion_matrix_data(ground_truth_entities, predictions, all_entity_types)
        all_model_results[model_name] = results
    
    # Print detailed results
    print_detailed_confusion_matrices(all_model_results)
    
    print("\n📈 GENERATING VISUALIZATIONS...")
    print("(Saving charts as PNG files)")
    
    # Create heatmaps for different metrics
    for metric in ['f1', 'precision', 'recall']:
        print(f"   Creating {metric.upper()} heatmap...")
        create_performance_heatmap(all_model_results, all_entity_types, metric)
    
    # Create comparison charts
    print("   Creating model comparison charts...")
    create_model_comparison_charts(all_model_results, all_entity_types)
    
    print("\n🎯 SUMMARY OF TRUTH VALUES:")
    print("=" * 50)
    
    for model_name, entity_results in all_model_results.items():
        print(f"\n{model_name.upper()}:")
        
        # Calculate overall confusion matrix
        total_tp = sum([r['tp'] for r in entity_results.values()])
        total_fp = sum([r['fp'] for r in entity_results.values()])
        total_tn = sum([r['tn'] for r in entity_results.values()])
        total_fn = sum([r['fn'] for r in entity_results.values()])
        
        print(f"  True Positives (Correct Detections):  {total_tp}")
        print(f"  False Positives (Wrong Detections):   {total_fp}")
        print(f"  True Negatives (Correct Rejections):  {total_tn}")
        print(f"  False Negatives (Missed Detections):  {total_fn}")
        
        # Performance summary
        total_predictions = total_tp + total_fp
        total_actual = total_tp + total_fn
        
        print(f"  Total Predictions Made: {total_predictions}")
        print(f"  Total Actual Entities:  {total_actual}")
        print(f"  Detection Rate: {total_tp/total_actual:.1%} of actual entities found")
        print(f"  Precision Rate: {total_tp/total_predictions:.1%} of predictions correct" if total_predictions > 0 else "  Precision Rate: N/A (no predictions)")
    
    print(f"\n📋 Analysis complete! Generated visualization files:")
    print("   - pii_performance_heatmap_f1.png")
    print("   - pii_performance_heatmap_precision.png") 
    print("   - pii_performance_heatmap_recall.png")
    print("   - model_performance_comparison.png")

if __name__ == "__main__":
    main()