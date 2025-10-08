import json

# Entities to ignore in FP counts (because Presidio does not detect them anyway)
IGNORE_FP_LABELS = {"CODE", "ORG", "QUANTITY", "MISC"}

def load_data(file_path="output_presidio_eval.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate(data):
    total_tp, total_fp, total_fn = 0, 0, 0

    for doc in data:
        eval_res = doc.get("evaluation", {})
        
        # Count True Positives
        total_tp += len(eval_res.get("true_positives", []))
        
        # Count False Negatives (missed detections)
        total_fn += len(eval_res.get("false_negatives", []))
        
        # Count False Positives, ignoring some labels
        for fp in eval_res.get("false_positives", []):
            if fp["label"] not in IGNORE_FP_LABELS:
                total_fp += 1

    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    data = load_data("output_presidio_eval.json")
    results = evaluate(data)

    print("📊 Evaluation Results")
    print(f"True Positives : {results['true_positives']}")
    print(f"False Positives: {results['false_positives']} (ignoring {IGNORE_FP_LABELS})")
    print(f"False Negatives: {results['false_negatives']}")
    print(f"Precision      : {results['precision']:.4f}")
    print(f"Recall         : {results['recall']:.4f}")
    print(f"F1 Score       : {results['f1']:.4f}")
