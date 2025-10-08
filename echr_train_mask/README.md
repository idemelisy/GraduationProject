# PII Detection Evaluation Methodology

## Overview
This repository contains a comprehensive evaluation framework for Personal Identifiable Information (PII) detection using Microsoft Presidio. The project evolved through multiple phases to address real-world challenges in NLP evaluation, particularly focusing on span matching, entity label ambiguity, and annotation quality assessment.

## Dataset
- **Source**: European Court of Human Rights (ECHR) legal documents
- **Format**: 1,014 documents with manual annotations
- **Annotation Types**: PERSON, LOCATION/LOC, DATETIME, DEM (Demographics), ORG
- **Ground Truth**: `annotations.json` with multiple annotators (67 multi-annotator documents)

## Project Evolution

### Phase 1: Initial Problem Identification
**Files**: `presidio_script.py`, `output_presidio_eval.json`

**Method**: Basic Presidio PII detection with exact span matching
- Applied Presidio Analyzer to detect PII entities
- Used exact character position matching for evaluation
- Generated initial detection results

**Issue Discovered**: 
- Henrik Hasslund appearing in both detected entities and false positives
- Root cause: Exact span mismatch (253-271 vs 256-271 character positions)
- High false positive rates due to minor span boundary differences

---

### Phase 2: Enhanced Evaluation Framework
**File**: `new_eval.py`

**Method**: IoU-based span matching with configurable threshold
- **Innovation**: Replaced exact character span matching with Intersection over Union (IoU ≥ 0.3)
- **Algorithm**: 
  ```python
  def iou_span(a_start, a_end, b_start, b_end):
      intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
      union = (a_end - a_start) + (b_end - b_start) - intersection
      return intersection / union if union > 0 else 0.0
  ```

**Key Features**:
- Entity label mapping: `LOCATION→LOC`, `DATE_TIME→DATETIME`, `NRP→DEM`
- Ignore filter for non-PII labels: `{CODE, ORG, QUANTITY, MISC}`
- Asymmetric ignore logic for both false positives and false negatives

**Output**: `detailed_eval_cumulative_fixed.json`

---

### Phase 3: Compound Entity Recognition
**Enhancement**: Cumulative IoU logic in `new_eval.py`

**Method**: `evaluate_document_cumulative_iou()` function
- **Purpose**: Handle cases where single ground truth spans are covered by multiple Presidio detections
- **Algorithm**: 
  ```python
  # Sum intersections from multiple predictions for single ground truth
  total_intersection = sum(intersections_with_gt)
  coverage = total_intersection / gt_length
  is_match = coverage >= iou_threshold
  ```

**Results**: 
- 6 compound entity matches found
- F1 improved from 0.7180 → 0.8158

---

### Phase 4: ORG-LOC Semantic Equivalence
**File**: `new_eval_relaxed_org_loc.py`

**Method**: Relaxed label matching with semantic equivalence groups
- **Innovation**: `are_labels_compatible()` function allowing ORG ↔ LOC cross-matching
- **Rationale**: Geographical entities can be legitimately annotated as either organizations or locations
- **Implementation**: 
  ```python
  ORG_LOC_EQUIVALENCE = {"ORG", "LOC"}
  
  def are_labels_compatible(gt_label, pred_label):
      if gt_label == pred_label:
          return True
      if gt_label in ORG_LOC_EQUIVALENCE and pred_label in ORG_LOC_EQUIVALENCE:
          return True
      return False
  ```

**Output**: `detailed_eval_relaxed_org_loc.json`

---

### Phase 5: Comparative Analysis
**Files**: `detailed_eval_strict.json` vs `detailed_eval_relaxed_org_loc.json`

**Method**: Side-by-side metric comparison between strict and relaxed matching
- Quantified improvement from addressing semantic label ambiguity
- Documented 2,270 relaxed ORG-LOC matches that improved evaluation accuracy

## Results Evolution

### Baseline (Initial)
```
Method: Basic exact span matching
F1 Score: ~0.72 (estimated from early runs)
Issues: High false positive rate due to span mismatches
```

### IoU-Based Matching (Strict)
```
File: detailed_eval_strict.json
Method: IoU threshold ≥ 0.3 + cumulative span coverage
True Positives: 47,735
False Positives: 10,044  
False Negatives: 11,509
Precision: 0.8262
Recall: 0.8057
F1 Score: 0.8158
```

### Relaxed ORG-LOC Matching (Final)
```
File: detailed_eval_relaxed_org_loc.json
Method: IoU ≥ 0.3 + ORG↔LOC equivalence
True Positives: 49,993 (+2,258)
False Positives: 7,679 (-2,365)
False Negatives: 11,521 (+12)
Precision: 0.8669 (+4.93%)
Recall: 0.8127 (+0.87%)
F1 Score: 0.8389 (+2.83%)
Relaxed Matches: 2,270
```

## Technical Methodologies

### 1. Intersection over Union (IoU) Span Matching
Replaces exact character position matching with overlap-based similarity:
- More robust to minor annotation differences
- Configurable threshold (default: 0.3)
- Handles both partial and complete overlaps

### 2. Cumulative Coverage Algorithm
Addresses compound entity detection scenarios:
- Single ground truth entity covered by multiple predictions
- Sums intersection coverage from all contributing predictions
- Marks all contributing predictions as "used" to prevent double-counting

### 3. Semantic Label Equivalence
Recognizes annotation ambiguity in geographical entities:
- "Republic of Turkey" can be ORG (political entity) or LOC (geographical location)
- "Ankara" can be LOC (city) or ORG (seat of government)
- Reduces evaluation noise from legitimate semantic variations

### 4. Entity Label Normalization
Maps Presidio output labels to ground truth schema:
```python
ENTITY_MAPPING = {
    "DATE_TIME": "DATETIME",
    "NRP": "DEM", 
    "PERSON": "PERSON",
    "LOC": "LOC",
    "LOCATION": "LOC"
}
```

## File Inventory

### Core Scripts
- `presidio_script.py` - Original Presidio detection pipeline
- `new_eval.py` - Enhanced evaluation with IoU matching
- `new_eval_relaxed_org_loc.py` - Relaxed ORG-LOC matching evaluation

### Data Files
- `output_presidio_eval.json` - Raw Presidio detection results
- `annotations.json` - Ground truth annotations (1,014 documents)
- `echr_train.xlsx` - Original ECHR dataset

### Evaluation Results
- `detailed_eval_cumulative_fixed.json` - IoU-based strict matching results
- `detailed_eval_strict.json` - Strict matching comparison baseline  
- `detailed_eval_relaxed_org_loc.json` - Relaxed ORG-LOC matching results

### Supporting Files
- `requirements.txt` - Python dependencies
- `metrics_presidio.py` - Additional metrics calculations

## Usage

### Basic Presidio Detection
```bash
python presidio_script.py
```

### Strict IoU-based Evaluation
```bash
python new_eval.py --input output_presidio_eval.json --out_detailed detailed_eval_strict.json --iou_threshold 0.3
```

### Relaxed ORG-LOC Evaluation
```bash
python new_eval_relaxed_org_loc.py --input output_presidio_eval.json --out_detailed detailed_eval_relaxed_org_loc.json --iou_threshold 0.3
```

## Key Findings

### 1. Annotation Quality Issues
- Discovered missing annotations for administrative dates (e.g., "1 November 2001")
- Context analysis revealed legitimate non-PII classifications for procedural dates
- Identified incorrect demographic labels for life events ("gave birth to a son")

### 2. Evaluation Methodology Impact
- IoU-based matching reduced false positives from span boundary issues
- Relaxed ORG-LOC matching addressed 2,270 semantic ambiguity cases
- F1 score improvement of 2.83% through methodological enhancements

### 3. Multi-Annotator Analysis
- 67 documents (6.6%) have multiple annotators
- 947 documents (93.4%) have single annotator
- Multi-annotator union strategy potential for future improvement

## Academic Contributions

1. **Multi-threshold IoU Evaluation**: More realistic than exact span matching
2. **Cumulative Span Coverage**: Handles compound entity detection scenarios  
3. **Semantic Label Equivalence**: Addresses annotation ambiguity in geographical entities
4. **Two-stage Evaluation Framework**: Strict vs. relaxed matching for comprehensive analysis
5. **Systematic Error Analysis**: Identified annotation quality issues vs. actual detection failures

## Dependencies

```txt
presidio-analyzer
presidio-anonymizer
pandas
numpy
json
argparse
```

## Authors
- [Your Name]
- [Institution]
- [Contact Information]

## License
[Specify your license]

## Citation
If you use this evaluation framework in your research, please cite:
```
[Your citation format]
```

---

This evaluation methodology demonstrates a thorough approach to PII detection assessment that addresses real-world annotation challenges and provides more meaningful performance metrics for production systems.