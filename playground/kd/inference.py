import os
import json
import argparse
#
import mlx.core as mx
#
from model import load_student
from generation import MAX_GEN_TOKENS, generate_json, extract_json, to_schema
from prompt import get_bbox_coord
#

# We have already merged the model with Qwen3-VL-2B-Thinking
KD_MODEL_ID = os.path.join(".", "models", "kd_model_merged")

# 
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
# 

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type = str, required = True, help = "image file to run on")
parser.add_argument("-m", "--model", type = str, default = KD_MODEL_ID, help = "path to the model")
parser.add_argument("--max-gen-tokens", type = int, default = MAX_GEN_TOKENS, help = "max tokens for the generated JSON")
args = parser.parse_args()

#
if __name__ == "__main__":
    student, processor = load_student(model_id = args.model)
    # 
    if not os.path.exists(args.image):
        print("Error, cannot find image")
        quit()
    # 
    bbox = get_bbox_coord(args.image)
    if bbox == "[]":
        print(f"OCR found no text lines in {args.image}, the student gets no bboxes")
    # 
    text = generate_json(student, processor, args.image, args.max_gen_tokens)
    #
    pred = extract_json(text)
    if pred is None:
        print(f"No valid JSON in the output for {args.image}, using an empty schema")
        pred = {}
    # 
    pred = to_schema(pred)

    print(f"\n{args.image}\n{json.dumps(pred, ensure_ascii = False, indent = 4)}")
    mx.clear_cache()