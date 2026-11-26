import os
import argparse
#
import mlx.core as mx
from tqdm import tqdm
#
from model import STUDENT_ID, load_student
from generation import MAX_GEN_TOKENS, generate_json, save_json
from data_generator import load_records, RECORD_TEST
#

JSON_OUT_DIR = os.path.join("..", "..", "data", "json_output", "kd_student")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--student", type = str, default = STUDENT_ID, help = "path to the student model")
parser.add_argument("-w", "--weights", type = str, default = None, help = "optional student.safetensors from main.py (the KD student)")
parser.add_argument("-o", "--out", type = str, default = JSON_OUT_DIR, help = "path to save the generated student JSON")
parser.add_argument("-r", "--records", type = str, default = RECORD_TEST, help = "record file of the images to run on")
parser.add_argument("--max-gen-tokens", type = int, default = MAX_GEN_TOKENS, help = "max tokens for the generated JSON")
args = parser.parse_args()

#
if __name__ == "__main__":
    student, processor = load_student(model_id = args.student)

    # main.py only saves the trainable weights, so they are loaded on
    # top of the base student model
    if args.weights is not None:
        student.load_weights(args.weights, strict = False)
    student.eval()

    # Loading the records
    records = load_records(args.records)

    for rec in tqdm(records):
        text = generate_json(student, processor, rec["image"], args.max_gen_tokens)
        save_json(rec["teacher_key"], text, args.out)
        mx.clear_cache()
