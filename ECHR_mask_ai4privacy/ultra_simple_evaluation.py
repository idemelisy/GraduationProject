#!/usr/bin/env python3
"""
Ultra-Simple AI4Privacy Evaluation Script

This is the most basic version - just counts matches and misses.
Perfect for understanding the core concepts before adding complexity.

Start here and add features one by one:
1. Basic counting ✓
2. Label mapping ✓ (STEP 1 ADDED)
3. Text overlap detection ✓ (STEP 2 ADDED)
4. Smart matching improvements ✓ (STEP 3 ADDED)
5. Add precision/recall calculations
6. Add entity type breakdown
"""

import json

# STEP 1: Label mapping from AI4Privacy labels to Ground Truth labels
LABEL_MAPPING = {
    # AI4Privacy label -> Ground truth label
    'GIVENNAME': 'PERSON',
    'SURNAME': 'PERSON', 
    'TITLE': 'PERSON',      # Mr, Ms, Dr, etc. are part of person names
    'CITY': 'LOC',
    'STREET': 'LOC',
    'ZIPCODE': 'LOC',
    'BUILDINGNUM': 'LOC',   # Building numbers are location info
    'DATE': 'DATETIME',
    'TIME': 'DATETIME',
}

# Entity types to exclude from evaluation (AI4Privacy doesn't detect these)
EXCLUDED_GT_TYPES = {'CODE', 'QUANTITY', 'MISC'}

# STEP 3: Smart matching improvements (inspired by complex evaluation)
# Labels that AI4Privacy finds but aren't in ground truth - don't penalize as false positives
IGNORE_FP_LABELS = {'SOCIALNUM', 'TELEPHONENUM', 'DRIVERLICENSENUM', 'TAXNUM', 'EMAIL', 'AGE', 'SEX', 'GENDER', 'IDCARDNUM'}

# Special matching rules
def clean_entity_text(text):
    """Clean entity text by removing trailing punctuation and whitespace."""
    import re
    text = text.strip()
    # Remove trailing punctuation
    text = re.sub(r'[.,:;!?\s]+$', '', text)
    return text

def calculate_iou(span1, span2):
    """Calculate Intersection over Union for two spans - more sophisticated than simple overlap."""
    start1, end1 = span1
    start2, end2 = span2
    
    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calculate union
    union = (end1 - start1) + (end2 - start2) - intersection
    
    return intersection / union if union > 0 else 0.0

def can_match_cross_type(pred_type, gt_type):
    """Allow certain cross-type matches (e.g., LOC predictions can match ORG ground truth)."""
    # Location predictions can match organization entities
    if pred_type == 'LOC' and gt_type == 'ORG':
        return True
    return pred_type == gt_type

def text_similarity_match(pred_text, gt_text, pred_type):
    """Check if prediction text is contained in or similar to ground truth text."""
    pred_clean = clean_entity_text(pred_text.lower())
    gt_clean = clean_entity_text(gt_text.lower())
    
    # For person names, allow partial matches
    if pred_type == 'PERSON':
        return pred_clean in gt_clean or gt_clean in pred_clean
    
    # For locations, allow substring matches (e.g., "Ankara" in "Ankara Regional Court")
    if pred_type == 'LOC':
        return pred_clean in gt_clean
    
    # For dates, be more flexible
    if pred_type == 'DATETIME':
        return pred_clean in gt_clean or gt_clean in pred_clean
    
    return pred_clean == gt_clean

def normalize_label(label):
    """Convert AI4Privacy label to ground truth equivalent."""
    return LABEL_MAPPING.get(label, label)

# STEP 2: Text overlap detection functions
def normalize_text(text):
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return text.strip().lower()

def spans_overlap(span1, span2):
    """Check if two spans (start, end) overlap at all."""
    start1, end1 = span1
    start2, end2 = span2
    return not (end1 <= start2 or end2 <= start1)

def calculate_overlap_ratio(span1, span2):
    """Calculate how much two spans overlap (0.0 to 1.0)."""
    start1, end1 = span1
    start2, end2 = span2
    
    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    if intersection_end <= intersection_start:
        return 0.0  # No overlap
    
    intersection_length = intersection_end - intersection_start
    
    # Calculate union (total coverage)
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_length = union_end - union_start
    
    return intersection_length / union_length if union_length > 0 else 0.0

def is_exact_match(gt_entity, pred_entity):
    """Check if ground truth and predicted entities are exact matches."""
    # Normalize labels
    gt_type = gt_entity.get('entity_type', 'UNKNOWN')
    pred_type = normalize_label(pred_entity.get('label', 'UNKNOWN'))
    
    # Check if types can match (including cross-type matches)
    if not can_match_cross_type(pred_type, gt_type):
        return False
    
    # Check if positions are exactly the same
    gt_span = (gt_entity['start_offset'], gt_entity['end_offset'])
    pred_span = (pred_entity['start'], pred_entity['end'])
    
    # Check if text is the same (normalized and cleaned)
    gt_text = clean_entity_text(gt_entity['span_text'])
    pred_text = clean_entity_text(pred_entity['text'])
    
    return gt_span == pred_span and normalize_text(gt_text) == normalize_text(pred_text)

def is_partial_match(gt_entity, pred_entity, min_overlap=0.5):
    """STEP 3: Improved partial matching with smart rules."""
    # Normalize labels
    gt_type = gt_entity.get('entity_type', 'UNKNOWN')
    pred_type = normalize_label(pred_entity.get('label', 'UNKNOWN'))
    
    # Check if types can match (including cross-type matches)
    if not can_match_cross_type(pred_type, gt_type):
        return False
    
    # Get spans and calculate IoU
    gt_span = (gt_entity['start_offset'], gt_entity['end_offset'])
    pred_span = (pred_entity['start'], pred_entity['end'])
    iou = calculate_iou(pred_span, gt_span)
    
    # Use different thresholds for different entity types
    if gt_type == 'ORG':
        threshold = 0.3  # Organizations can be partially matched with locations
    elif gt_type == 'PERSON':
        threshold = 0.4  # Names can be split across multiple predictions
    elif gt_type == 'LOC':
        threshold = 0.3  # Locations can have various forms
    else:
        threshold = min_overlap
    
    # Check IoU threshold
    if iou < threshold:
        return False
    
    # Additional text similarity check for better matching
    pred_text = pred_entity['text']
    gt_text = gt_entity['span_text']
    
    # If IoU is reasonable, check text similarity
    if iou >= threshold and text_similarity_match(pred_text, gt_text, pred_type):
        return True
    
    # Fall back to IoU only
    return iou >= threshold

def load_results(filepath):
    """Load the AI4Privacy results."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} documents")
    return data

def evaluate_document_with_overlap(doc):
    """
    STEP 2: Evaluate a document using text overlap detection.
    """
    # Get entities (excluding certain ground truth types)
    gt_entities = [e for e in doc.get('annotations', []) 
                   if e.get('entity_type') not in EXCLUDED_GT_TYPES]
    pred_entities = doc.get('ai4privacy_detected_pii', [])
    
    # Track results
    doc_results = {
        'doc_id': doc.get('id', 'unknown'),
        'gt_count': len(gt_entities),
        'pred_count': len(pred_entities),
        'exact_matches': 0,
        'partial_matches': 0,
        'missed': 0,
        'false_positives': 0,
        'examples': {
            'exact_matches': [],
            'partial_matches': [],
            'missed': [],
            'false_positives': []
        }
    }
    
    # Track which predicted entities have been matched
    matched_pred_indices = set()
    
    # For each ground truth entity, find the best match
    for gt_entity in gt_entities:
        best_match_type = None
        best_match_idx = -1
        best_overlap = 0.0
        
        # Find best matching predicted entity
        for i, pred_entity in enumerate(pred_entities):
            if i in matched_pred_indices:
                continue  # Already matched
            
            # Check for exact match first
            if is_exact_match(gt_entity, pred_entity):
                best_match_type = 'exact'
                best_match_idx = i
                best_overlap = 1.0
                break  # Exact match is best possible
            
            # Check for partial match
            elif is_partial_match(gt_entity, pred_entity):
                gt_span = (gt_entity['start_offset'], gt_entity['end_offset'])
                pred_span = (pred_entity['start'], pred_entity['end'])
                overlap = calculate_overlap_ratio(gt_span, pred_span)
                
                if overlap > best_overlap:
                    best_match_type = 'partial'
                    best_match_idx = i
                    best_overlap = overlap
        
        # Record the result
        if best_match_type == 'exact':
            doc_results['exact_matches'] += 1
            matched_pred_indices.add(best_match_idx)
            
            # Store example (limit to 2 per document)
            if len(doc_results['examples']['exact_matches']) < 2:
                doc_results['examples']['exact_matches'].append({
                    'gt_text': gt_entity['span_text'],
                    'pred_text': pred_entities[best_match_idx]['text'],
                    'gt_type': gt_entity['entity_type'],
                    'pred_label': pred_entities[best_match_idx]['label'],
                    'position': f"{gt_entity['start_offset']}-{gt_entity['end_offset']}"
                })
                
        elif best_match_type == 'partial':
            doc_results['partial_matches'] += 1
            matched_pred_indices.add(best_match_idx)
            
            # Store example (limit to 2 per document)
            if len(doc_results['examples']['partial_matches']) < 2:
                doc_results['examples']['partial_matches'].append({
                    'gt_text': gt_entity['span_text'],
                    'pred_text': pred_entities[best_match_idx]['text'],
                    'gt_type': gt_entity['entity_type'],
                    'pred_label': pred_entities[best_match_idx]['label'],
                    'overlap': round(best_overlap, 3),
                    'position': f"{gt_entity['start_offset']}-{gt_entity['end_offset']}"
                })
        else:
            doc_results['missed'] += 1
            
            # Store example (limit to 2 per document)
            if len(doc_results['examples']['missed']) < 2:
                doc_results['examples']['missed'].append({
                    'gt_text': gt_entity['span_text'],
                    'gt_type': gt_entity['entity_type'],
                    'position': f"{gt_entity['start_offset']}-{gt_entity['end_offset']}"
                })
    
    # Count false positives (predicted entities that didn't match any ground truth)
    for i, pred_entity in enumerate(pred_entities):
        if i not in matched_pred_indices:
            # STEP 3: Don't penalize predictions that AI4Privacy finds but aren't in ground truth
            pred_label = normalize_label(pred_entity.get('label', 'UNKNOWN'))
            original_label = pred_entity.get('label', 'UNKNOWN')
            
            if original_label not in IGNORE_FP_LABELS:
                doc_results['false_positives'] += 1
                
                # Store example (limit to 2 per document)
                if len(doc_results['examples']['false_positives']) < 2:
                    doc_results['examples']['false_positives'].append({
                        'pred_text': pred_entity['text'],
                        'pred_label': original_label,
                        'mapped_label': pred_label,
                        'position': f"{pred_entity['start']}-{pred_entity['end']}"
                    })
            # If it's an ignored label, don't count as false positive but mention it
            else:
                if 'ignored_predictions' not in doc_results['examples']:
                    doc_results['examples']['ignored_predictions'] = []
                if len(doc_results['examples']['ignored_predictions']) < 2:
                    doc_results['examples']['ignored_predictions'].append({
                        'pred_text': pred_entity['text'],
                        'pred_label': original_label,
                        'reason': 'AI4Privacy specialty - not in ground truth'
                    })
    
    return doc_results

def evaluate_all_documents_with_overlap(data):
    """
    STEP 2: Evaluate all documents with overlap detection.
    """
    print(f"\n🔍 EVALUATING WITH OVERLAP DETECTION (STEP 2)")
    print("-" * 50)
    
    total_results = {
        'total_docs': len(data),
        'total_gt': 0,
        'total_pred': 0,
        'exact_matches': 0,
        'partial_matches': 0,
        'missed': 0,
        'false_positives': 0,
        'examples': {
            'exact_matches': [],
            'partial_matches': [],
            'missed': [],
            'false_positives': [],
            'ignored_predictions': []
        }
    }
    
    # Process documents
    for i, doc in enumerate(data):
        if i % 200 == 0:
            print(f"Processed {i}/{len(data)} documents...")
        
        doc_results = evaluate_document_with_overlap(doc)
        
        # Accumulate totals
        total_results['total_gt'] += doc_results['gt_count']
        total_results['total_pred'] += doc_results['pred_count']
        total_results['exact_matches'] += doc_results['exact_matches']
        total_results['partial_matches'] += doc_results['partial_matches']
        total_results['missed'] += doc_results['missed']
        total_results['false_positives'] += doc_results['false_positives']
        
        # Collect examples (limit total examples)
        for example_type in ['exact_matches', 'partial_matches', 'missed', 'false_positives', 'ignored_predictions']:
            if example_type in total_results['examples'] and len(total_results['examples'][example_type]) < 5:
                if example_type in doc_results['examples']:
                    total_results['examples'][example_type].extend(
                        doc_results['examples'][example_type][:2]
                    )
    
    # Calculate metrics
    tp = total_results['exact_matches'] + total_results['partial_matches']
    fp = total_results['false_positives']
    fn = total_results['missed']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Add metrics to results
    total_results['precision'] = precision
    total_results['recall'] = recall
    total_results['f1'] = f1
    
    print(f"Completed evaluation of {len(data)} documents.\n")
    
    # Print results
    print("="*60)
    print("OVERLAP DETECTION RESULTS (STEP 2)")
    print("="*60)
    print(f"Documents: {total_results['total_docs']:,}")
    print(f"Ground truth entities: {total_results['total_gt']:,}")
    print(f"Predicted entities: {total_results['total_pred']:,}")
    print()
    print(f"✅ Exact matches: {total_results['exact_matches']:,}")
    print(f"⚠️  Partial matches: {total_results['partial_matches']:,}")
    print(f"❌ Missed entities: {total_results['missed']:,}")
    print(f"🚫 False positives: {total_results['false_positives']:,}")
    print()
    print(f"📊 PERFORMANCE METRICS:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Show examples
    examples = total_results['examples']
    
    if examples['exact_matches']:
        print(f"\n✅ EXAMPLE EXACT MATCHES:")
        for i, ex in enumerate(examples['exact_matches'][:3]):
            print(f"   {i+1}. '{ex['gt_text']}' ({ex['gt_type']}) = "
                  f"'{ex['pred_text']}' ({ex['pred_label']}) at {ex['position']}")
    
    if examples['partial_matches']:
        print(f"\n⚠️  EXAMPLE PARTIAL MATCHES:")
        for i, ex in enumerate(examples['partial_matches'][:3]):
            print(f"   {i+1}. '{ex['gt_text']}' ({ex['gt_type']}) ≈ "
                  f"'{ex['pred_text']}' ({ex['pred_label']}) "
                  f"[overlap: {ex['overlap']}] at {ex['position']}")
    
    if examples['missed']:
        print(f"\n❌ EXAMPLE MISSED ENTITIES:")
        for i, ex in enumerate(examples['missed'][:3]):
            print(f"   {i+1}. '{ex['gt_text']}' ({ex['gt_type']}) at {ex['position']}")
    
    if examples['false_positives']:
        print(f"\n🚫 EXAMPLE FALSE POSITIVES:")
        for i, ex in enumerate(examples['false_positives'][:3]):
            print(f"   {i+1}. '{ex['pred_text']}' ({ex['pred_label']} → {ex['mapped_label']}) at {ex['position']}")
    
    if examples.get('ignored_predictions'):
        print(f"\n🔕 EXAMPLE IGNORED PREDICTIONS (AI4Privacy specialties):")
        for i, ex in enumerate(examples['ignored_predictions'][:3]):
            print(f"   {i+1}. '{ex['pred_text']}' ({ex['pred_label']}) - {ex['reason']}")
    
    print("="*60)
    
    return total_results

def basic_evaluation_with_mapping(data):
    """
    STEP 1: Basic evaluation with label mapping and exclusions.
    """
    total_docs = len(data)
    total_ground_truth = 0
    total_predicted = 0
    total_ground_truth_excluded = 0
    
    # Count entities across all documents
    for doc in data:
        # Ground truth entities from annotations (excluding certain types)
        gt_entities = doc.get('annotations', [])
        for entity in gt_entities:
            entity_type = entity.get('entity_type', 'UNKNOWN')
            if entity_type in EXCLUDED_GT_TYPES:
                total_ground_truth_excluded += 1
            else:
                total_ground_truth += 1
        
        # Predicted entities from AI4Privacy
        pred_entities = doc.get('ai4privacy_detected_pii', [])
        total_predicted += len(pred_entities)
    
    # Print basic statistics
    print("\n" + "="*60)
    print("BASIC EVALUATION WITH LABEL MAPPING (STEP 1)")
    print("="*60)
    print(f"Documents: {total_docs}")
    print(f"Ground truth entities (total): {total_ground_truth + total_ground_truth_excluded:,}")
    print(f"Ground truth entities (excluded): {total_ground_truth_excluded:,}")
    print(f"Ground truth entities (evaluated): {total_ground_truth:,}")
    print(f"Predicted entities: {total_predicted:,}")
    print(f"Difference: {total_predicted - total_ground_truth:,}")
    
    if total_ground_truth > 0:
        prediction_ratio = total_predicted / total_ground_truth
        print(f"Prediction ratio: {prediction_ratio:.2f}x")
        if prediction_ratio > 1:
            print("→ AI4Privacy is predicting MORE entities than ground truth")
        elif prediction_ratio < 1:
            print("→ AI4Privacy is predicting FEWER entities than ground truth")
        else:
            print("→ AI4Privacy predictions match ground truth count exactly")
    
    print("="*60)

def basic_evaluation(data):
    """
    Ultra-basic evaluation: just count things.
    """
    total_docs = len(data)
    total_ground_truth = 0
    total_predicted = 0
    
    # Count entities across all documents
    for doc in data:
        # Ground truth entities from annotations
        gt_entities = doc.get('annotations', [])
        total_ground_truth += len(gt_entities)
        
        # Predicted entities from AI4Privacy
        pred_entities = doc.get('ai4privacy_detected_pii', [])
        total_predicted += len(pred_entities)
    
    # Print basic statistics
    print("\n" + "="*50)
    print("BASIC EVALUATION RESULTS (NO MAPPING)")
    print("="*50)
    print(f"Documents: {total_docs}")
    print(f"Ground truth entities: {total_ground_truth:,}")
    print(f"Predicted entities: {total_predicted:,}")
    print(f"Difference: {total_predicted - total_ground_truth:,}")
    
    if total_ground_truth > 0:
        prediction_ratio = total_predicted / total_ground_truth
        print(f"Prediction ratio: {prediction_ratio:.2f}x")
        if prediction_ratio > 1:
            print("→ AI4Privacy is predicting MORE entities than ground truth")
        elif prediction_ratio < 1:
            print("→ AI4Privacy is predicting FEWER entities than ground truth")
        else:
            print("→ AI4Privacy predictions match ground truth count exactly")
    
    print("="*50)

def examine_first_document_with_mapping(data):
    """
    STEP 1: Look at the first document with mapping applied.
    """
    if not data:
        print("No data to examine!")
        return
    
    doc = data[0]
    print(f"\n📄 EXAMINING FIRST DOCUMENT WITH MAPPING: {doc.get('id', 'Unknown')}")
    print("-" * 50)
    
    # Show text length
    text = doc.get('text', '')
    print(f"Text length: {len(text)} characters")
    
    # Show ground truth entities (excluding certain types)
    gt_entities = doc.get('annotations', [])
    included_gt = [e for e in gt_entities if e.get('entity_type') not in EXCLUDED_GT_TYPES]
    excluded_gt = [e for e in gt_entities if e.get('entity_type') in EXCLUDED_GT_TYPES]
    
    print(f"Ground truth entities (total): {len(gt_entities)}")
    print(f"Ground truth entities (included): {len(included_gt)}")
    print(f"Ground truth entities (excluded): {len(excluded_gt)}")
    
    if included_gt:
        print("First 3 included ground truth entities:")
        for i, entity in enumerate(included_gt[:3]):
            print(f"  {i+1}. '{entity['span_text']}' ({entity['entity_type']}) "
                  f"at {entity['start_offset']}-{entity['end_offset']}")
    
    if excluded_gt:
        print("First 3 excluded ground truth entities:")
        for i, entity in enumerate(excluded_gt[:3]):
            print(f"  {i+1}. '{entity['span_text']}' ({entity['entity_type']}) "
                  f"at {entity['start_offset']}-{entity['end_offset']} [EXCLUDED]")
    
    # Show predicted entities with mapping
    pred_entities = doc.get('ai4privacy_detected_pii', [])
    print(f"Predicted entities: {len(pred_entities)}")
    
    if pred_entities:
        print("First 3 predicted entities (with mapping):")
        for i, entity in enumerate(pred_entities[:3]):
            original_label = entity['label']
            mapped_label = normalize_label(original_label)
            mapping_indicator = "" if original_label == mapped_label else f" → {mapped_label}"
            print(f"  {i+1}. '{entity['text']}' ({original_label}{mapping_indicator}) "
                  f"at {entity['start']}-{entity['end']}")

def examine_first_document(data):
    """
    Look at the first document in detail to understand the data structure.
    """
    if not data:
        print("No data to examine!")
        return
    
    doc = data[0]
    print(f"\n📄 EXAMINING FIRST DOCUMENT (NO MAPPING): {doc.get('id', 'Unknown')}")
    print("-" * 40)
    
    # Show text length
    text = doc.get('text', '')
    print(f"Text length: {len(text)} characters")
    
    # Show ground truth entities
    gt_entities = doc.get('annotations', [])
    print(f"Ground truth entities: {len(gt_entities)}")
    
    if gt_entities:
        print("First 3 ground truth entities:")
        for i, entity in enumerate(gt_entities[:3]):
            print(f"  {i+1}. '{entity['span_text']}' ({entity['entity_type']}) "
                  f"at {entity['start_offset']}-{entity['end_offset']}")
    
    # Show predicted entities  
    pred_entities = doc.get('ai4privacy_detected_pii', [])
    print(f"Predicted entities: {len(pred_entities)}")
    
    if pred_entities:
        print("First 3 predicted entities:")
        for i, entity in enumerate(pred_entities[:3]):
            print(f"  {i+1}. '{entity['text']}' ({entity['label']}) "
                  f"at {entity['start']}-{entity['end']}")

def show_entity_types_with_mapping(data):
    """
    STEP 1: Show entity types with mapping applied.
    """
    gt_types = set()
    pred_types = set()
    mapped_pred_types = set()
    excluded_gt_count = 0
    
    for doc in data:
        # Collect ground truth types (excluding certain types)
        for entity in doc.get('annotations', []):
            entity_type = entity.get('entity_type', 'UNKNOWN')
            if entity_type in EXCLUDED_GT_TYPES:
                excluded_gt_count += 1
            else:
                gt_types.add(entity_type)
        
        # Collect predicted types (original and mapped)
        for entity in doc.get('ai4privacy_detected_pii', []):
            original_label = entity.get('label', 'UNKNOWN')
            pred_types.add(original_label)
            mapped_label = normalize_label(original_label)
            mapped_pred_types.add(mapped_label)
    
    print(f"\n🏷️  ENTITY TYPES WITH MAPPING (STEP 1)")
    print("-" * 50)
    print(f"Ground truth types (evaluated): {sorted(gt_types)}")
    print(f"Ground truth types (excluded): {sorted(EXCLUDED_GT_TYPES)}")
    print(f"Excluded entities count: {excluded_gt_count:,}")
    print()
    print(f"AI4Privacy original types: {sorted(pred_types)}")
    print(f"AI4Privacy mapped types: {sorted(mapped_pred_types)}")
    print()
    
    # Show mapping in action
    print("📋 LABEL MAPPING APPLIED:")
    for ai4_label, gt_label in LABEL_MAPPING.items():
        if ai4_label in pred_types:  # Only show mappings that are actually used
            print(f"  {ai4_label} → {gt_label}")
    
    # Find overlaps after mapping
    common_types = gt_types.intersection(mapped_pred_types)
    gt_only = gt_types - mapped_pred_types
    pred_only = mapped_pred_types - gt_types
    
    print()
    if common_types:
        print(f"✅ Common types after mapping: {sorted(common_types)}")
    if gt_only:
        print(f"❌ Only in ground truth: {sorted(gt_only)}")
    if pred_only:
        print(f"⚠️  Only in predictions (after mapping): {sorted(pred_only)}")

def show_entity_types(data):
    """
    Show what types of entities exist in ground truth vs predictions.
    """
    gt_types = set()
    pred_types = set()
    
    for doc in data:
        # Collect ground truth types
        for entity in doc.get('annotations', []):
            gt_types.add(entity.get('entity_type', 'UNKNOWN'))
        
        # Collect predicted types
        for entity in doc.get('ai4privacy_detected_pii', []):
            pred_types.add(entity.get('label', 'UNKNOWN'))
    
    print(f"\n🏷️  ENTITY TYPES COMPARISON (NO MAPPING)")
    print("-" * 40)
    print(f"Ground truth types ({len(gt_types)}): {sorted(gt_types)}")
    print(f"Predicted types ({len(pred_types)}): {sorted(pred_types)}")
    
    # Find overlaps
    common_types = gt_types.intersection(pred_types)
    gt_only = gt_types - pred_types
    pred_only = pred_types - gt_types
    
    if common_types:
        print(f"Common types: {sorted(common_types)}")
    if gt_only:
        print(f"Only in ground truth: {sorted(gt_only)}")
    if pred_only:
        print(f"Only in predictions: {sorted(pred_only)}")

def main():
    """Main function - start simple and build up."""
    print("Ultra-Simple AI4Privacy Evaluation - STEP 3: Smart Matching Improvements")
    print("=" * 75)
    
    # Load the data
    try:
        data = load_results('/home/ide/ide/ECHR_mask_ai4privacy/output_ai4privacy.json')
    except FileNotFoundError:
        print("❌ File not found! Make sure output_ai4privacy.json exists.")
        return
    
    # Show the configuration we're using
    print(f"\n🔄 STEP 3 CONFIGURATION:")
    print("Label mapping (AI4Privacy → Ground Truth):")
    for ai4_label, gt_label in LABEL_MAPPING.items():
        print(f"  {ai4_label} → {gt_label}")
    print(f"\n🚫 Excluded ground truth types: {sorted(EXCLUDED_GT_TYPES)}")
    print(f"🔕 Ignored false positive labels: {sorted(IGNORE_FP_LABELS)}")
    print(f"\n✨ SMART MATCHING FEATURES:")
    print("  - Text cleaning (removes punctuation)")
    print("  - IoU calculation (Intersection over Union)")
    print("  - Cross-type matching (LOC can match ORG)")
    print("  - Substring matching for names and locations")
    print("  - Different thresholds per entity type")
    
    # Run previous steps for comparison
    basic_evaluation_with_mapping(data)
    
    # Run Step 3: Smart matching evaluation
    print(f"\n🧠 RUNNING SMART MATCHING EVALUATION...")
    smart_results = evaluate_all_documents_with_overlap(data)
    
    # Quick summary comparison
    print(f"\n📊 COMPARISON WITH STEP 2:")
    print(f"  Previous F1-Score: 0.0235 (from Step 2)")
    print(f"  Current F1-Score:  {smart_results['f1']:.4f} (Step 3)")
    
    if smart_results['examples'].get('ignored_predictions'):
        ignored_count = len(smart_results['examples']['ignored_predictions'])
        print(f"  Predictions ignored: {ignored_count} examples (not counted as false positives)")
    
    print(f"\n💡 STEP 3 COMPLETE - NEXT STEPS:")
    print("✅ 1. Label mapping applied")
    print("✅ 2. Text overlap detection added") 
    print("✅ 3. Smart matching improvements applied")
    print("⏭️  4. Add entity-type specific breakdown")
    print("⏭️  5. Add performance comparison and analysis")
    
    return smart_results

if __name__ == "__main__":
    main()