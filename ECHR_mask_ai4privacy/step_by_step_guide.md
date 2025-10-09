# AI4Privacy Evaluation - Step by Step Guide

This guide shows you how to build an evaluation script from scratch, adding one concept at a time.

## 📁 Files Created

1. **`ultra_simple_evaluation.py`** - The most basic version (just counting)
2. **`simple_evaluation.py`** - More complete version with precision/recall
3. **`step_by_step_guide.md`** - This guide

## 🎯 Current Status

From the ultra-simple evaluation, we learned:

- **1,014 documents** in total
- **95,030 ground truth entities** (from human annotations)
- **109,136 predicted entities** (from AI4Privacy)
- **1.15x prediction ratio** - AI4Privacy predicts 15% more entities than ground truth

### Key Issues Discovered:

1. **No common entity types** between ground truth and predictions
   - Ground truth uses: `CODE`, `DATETIME`, `DEM`, `LOC`, `MISC`, `ORG`, `PERSON`, `QUANTITY`
   - AI4Privacy uses: `BUILDINGNUM`, `CITY`, `DATE`, `GIVENNAME`, `SURNAME`, `TITLE`, etc.

2. **Label mismatch problem** - This is why performance appears low
   - Ground truth `DATETIME` vs AI4Privacy `DATE` 
   - Ground truth `PERSON` vs AI4Privacy `GIVENNAME`/`SURNAME`/`TITLE`
   - Ground truth `LOC` vs AI4Privacy `CITY`/`STREET`

## 🛠️ Next Steps to Improve Evaluation

### Step 1: Add Label Mapping
Create a mapping between ground truth labels and AI4Privacy labels:

```python
LABEL_MAPPING = {
    # AI4Privacy label -> Ground truth label
    'DATE': 'DATETIME',
    'TIME': 'DATETIME', 
    'GIVENNAME': 'PERSON',
    'SURNAME': 'PERSON',
    'TITLE': 'PERSON',
    'CITY': 'LOC',
    'STREET': 'LOC',
    'BUILDINGNUM': 'CODE',  # or 'QUANTITY' depending on context
    # Add more mappings as needed
}
```

### Step 2: Add Basic Precision/Recall
```python
def calculate_precision_recall(true_positives, false_positives, false_negatives):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1
```

### Step 3: Add Text Overlap Detection
```python
def spans_overlap(span1, span2, min_overlap=0.5):
    start1, end1 = span1
    start2, end2 = span2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_end <= overlap_start:
        return False
    overlap_length = overlap_end - overlap_start
    total_length = min(end1 - start1, end2 - start2)
    return (overlap_length / total_length) >= min_overlap
```

### Step 4: Add Entity-Type Specific Metrics
Track performance separately for each entity type to see which types AI4Privacy handles well.

### Step 5: Add Example Collection
Collect examples of:
- Perfect matches
- Partial matches  
- Missed entities
- False positives

## 🔧 How to Extend the Ultra-Simple Script

### Adding Label Mapping (Easy)
1. Open `ultra_simple_evaluation.py`
2. Add the `LABEL_MAPPING` dictionary at the top
3. Create a function to normalize labels:
```python
def normalize_label(label):
    return LABEL_MAPPING.get(label, label)
```

### Adding Precision/Recall (Medium)
1. Count matches properly by comparing normalized labels and text overlap
2. Calculate TP, FP, FN for each document
3. Sum up across all documents and calculate final metrics

### Adding Detailed Analysis (Advanced)
1. Track performance by entity type
2. Collect examples for manual inspection
3. Add statistical significance tests
4. Add visualization capabilities

## 🎯 Expected Performance After Fixes

Once you add label mapping, you should see much better performance because:
- `DATE` entities will match `DATETIME` entities
- `GIVENNAME`/`SURNAME` will match `PERSON` entities  
- `CITY` entities will match `LOC` entities

The simple evaluation (with mapping) shows around **F1: 0.047**, but this will improve significantly with proper label alignment.

## 📊 Understanding the Current Results

From `simple_evaluation.py` results:
- **Overall F1: 0.047** (very low due to label mismatch)
- **Best performing type: CODE** (F1: 0.685) - because some overlap exists
- **Worst: Most types** (F1: near 0) - due to label mismatch

## 🚀 Quick Win Strategy

1. **Start with `ultra_simple_evaluation.py`** - Understand the basics
2. **Add label mapping** - This will give you the biggest improvement
3. **Add overlap detection** - This will catch partial matches
4. **Add entity-type breakdown** - This will show where AI4Privacy excels/struggles
5. **Use `simple_evaluation.py`** as reference for advanced features

## 💡 Tips for Development

- **Test with small data first** - Use `data[:10]` to test changes quickly
- **Print intermediate results** - Add debug prints to understand what's happening
- **Save results to files** - So you can compare different approaches
- **Focus on one improvement at a time** - Don't try to add everything at once