import os
import json
import argparse
# 
from jiwer import cer, wer
from khmerns import tokenize, normalize
# 

PRED_DIR = os.path.join("..", "..", "data", "json_output", "teacher")
GOLD_DIR = os.path.join("..", "..", "data", "json_output", "test")

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gold_dir", type = str, default= GOLD_DIR, help="path to the golden labels")
parser.add_argument("-p", "--pred_dir", type = str, default= PRED_DIR, help="path to the pred labels")
args = parser.parse_args()

# For calculating average
count = 0

# 
def is_empty(val : any) -> bool:
    return val in ("", None, False)

def segment_word(word : str) -> list[str]:
    return tokenize(normalize(word))

def calculate_cer(pred : str, gold : str) -> float:
    return cer(gold, pred)

def calculate_wer(pred : str, gold : str) -> float:
    pred_tokens = segment_word(pred)
    gold_tokens = segment_word(gold)
    return wer(" ".join(gold_tokens), " ".join(pred_tokens))

def calculate_prec_rec_f1 (pred : dict, gold : dict) -> tuple[float, float, float]:
    tp, fp, fn = 0, 0, 0

    for key, val in gold.items():
        pred_val = pred.get(key)

        # Both empty: nothing to extract, skip
        if is_empty(val) and is_empty(pred_val):
            continue
        # Gold empty but pred filled: hallucinated value
        if is_empty(val):
            fp += 1
            continue
        # Gold filled but pred missing or empty: missed value
        if is_empty(pred_val):
            fn += 1
            continue
        # Both filled but different: wrong value is extracted, and
        # the correct one is missed
        if pred_val != val:
            fp += 1
            fn += 1
            continue
        tp += 1

    # Handles hallucination: keys that are not in the schema
    for key, val in pred.items():
        if key not in gold and not is_empty(val):
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return (precision, recall, f1)

def verify_accuracy (pred : str, gold : str):
    # This is just a binary calculation of whether the string matches
    # Returns 1 or 0 and is handled in the procedural logic
    return 1 if pred == gold else 0

# Running this procedurally to get the result
if __name__ == "__main__":
    if not os.path.exists(args.pred_dir):
        print("Path to the pred directory does not exist")
        exit()
    
    if not os.path.exists(args.gold_dir):
        print("Path to the gold directory does not exist")
        exit()

    # Instantiating the variables 
    cum_acc, cum_prec, cum_rec, cum_f1, cum_cer, cum_wer = 0, 0, 0, 0, 0, 0
    
    gold_files = os.listdir(args.gold_dir)
    pred_files = set(os.listdir(args.pred_dir))

    for fn in gold_files:
        filename = str(fn.split(".")[0])
        # Check if the json file exists in the prediction
        if fn not in pred_files:
            continue

        pred, gold = "", ""
        count += 1

        # Broken JSON or a non-dict output from the model counts as an
        # empty prediction (all fields missed) instead of crashing the run
        try:
            with open(os.path.join(args.pred_dir, fn), encoding="utf-8") as pd:
                pred = json.load(pd)
        except json.JSONDecodeError:
            print(f"Invalid JSON in prediction {fn}, scoring as empty")
            pred = {}

        if not isinstance(pred, dict):
            print(f"Prediction {fn} is not a JSON object, scoring as empty")
            pred = {}

        with open(os.path.join(args.gold_dir, fn), encoding="utf-8") as gd:
            gold = json.load(gd)

        # Calculating the cumalative metrics
        _p, _r, _f = calculate_prec_rec_f1(pred, gold)

        pred_str = json.dumps(pred, sort_keys = True, ensure_ascii = False)
        gold_str = json.dumps(gold, sort_keys = True, ensure_ascii = False)

        cum_acc += verify_accuracy(pred_str, gold_str)
        cum_cer += calculate_cer(pred_str, gold_str)
        cum_wer += calculate_wer(pred_str, gold_str)
        cum_prec += _p
        cum_rec += _r
        cum_f1 += _f

    # Check if count is 0
    if count == 0:
        print("No file was accessed, breaking")
        quit()

    # Writing to a metric json file
    with open(os.path.join(".", "metrics", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "acc": cum_acc / count,
                "cer": cum_cer / count,
                "wer": cum_wer / count,
                "prec": cum_prec / count,
                "rec": cum_rec / count,
                "f1": cum_f1 / count
            },
            f,
            indent=4,
        )