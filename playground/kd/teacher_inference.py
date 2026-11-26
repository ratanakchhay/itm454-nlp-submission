import os
import argparse
#
import mlx.core as mx
from tqdm import tqdm
#
from model import load_teacher, answer_logits
from generation import MAX_GEN_TOKENS, generate_json, save_json
from data_generator import DataGenerator, load_records, RECORD_TRAIN, RECORD_VAL, RECORD_TEST
from topk import TEACHER_DIR, TOP_K, save_teacher_topk
#

JSON_OUT_DIR = os.path.join("..", "..", "data", "json_output", "teacher")

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", type = str, default = TEACHER_DIR)
parser.add_argument("-k", "--k", type = int, default = TOP_K)
parser.add_argument("-m", "--max-answer-tokens", type = int, default = None)
parser.add_argument("-j", "--json-out", type = str, default = JSON_OUT_DIR, help = "path to save the generated teacher JSON")
parser.add_argument("--max-gen-tokens", type = int, default = MAX_GEN_TOKENS, help = "max tokens for the generated JSON")
parser.add_argument("--topk-records", type = str, nargs = "*", default = [RECORD_TRAIN, RECORD_VAL], help = "record files to save the top-K logits for")
parser.add_argument("--gen-records", type = str, nargs = "*", default = [RECORD_TEST], help = "record files to generate the JSON for")
args = parser.parse_args()

#
if __name__ == "__main__":
    teacher, processor = load_teacher()

    # Top-K logits of the gold answer (teacher-forced)
    for path in args.topk_records:
        records = load_records(path)
        dataset = DataGenerator(records, processor, teacher_dir = args.out, max_answer_tokens = args.max_answer_tokens)

        for idx in tqdm(range(len(dataset)), desc = f"top-k {os.path.basename(path)}"):
            start, n = dataset.answer_span(idx)
            enc = dataset.encode(idx)
            inputs = {
                "input_ids": mx.array(enc["input_ids"][None].astype("int32")),
                "pixel_values": mx.array(enc["pixel_values"]),
                "image_grid_thw": mx.array(enc["image_grid_thw"]),
            }

            # Extracting just the answer logits
            logits = answer_logits(teacher, inputs, start - 1, start - 1 + n)
            mx.eval(logits)
            save_teacher_topk(records[idx]["teacher_key"], logits, args.k, args.out)

            # Clearing cache to free up memory
            del logits, inputs
            mx.clear_cache()

    # Generating the teacher's own JSON output
    for path in args.gen_records:
        for rec in tqdm(load_records(path), desc = f"generate {os.path.basename(path)}"):
            text = generate_json(teacher, processor, rec["image"], args.max_gen_tokens)
            save_json(rec["teacher_key"], text, args.json_out)
            # Clearing cache to free up memory
            mx.clear_cache()
