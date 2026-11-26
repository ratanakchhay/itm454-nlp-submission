import os
import json
import random
# 
import numpy as np
import mlx.core as mx
from PIL import Image
# 
from prompt import build_chat_prompt, END_TOKEN
from topk import TEACHER_DIR, load_teacher_topk
#

#

# Produced by split.py + records.py (one record file per split)
RECORD_TRAIN = os.path.join("..", "..", "data", "record_train.jsonl")
RECORD_VAL = os.path.join("..", "..", "data", "record_val.jsonl")
RECORD_TEST = os.path.join("..", "..", "data", "record_test.jsonl")

#

def load_records(path = RECORD_TRAIN):
    # The image file name (without extension) is the teacher file name
    # (data/teacher/<image_name>.npz), so records can be reordered or split freely.
    records = []
    with open(path, encoding = "utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["image"] = os.path.join("..", "..", rec["image"]) # record paths are relative to the repo root
            rec["teacher_key"] = os.path.splitext(os.path.basename(rec["image"]))[0]
            records.append(rec)
    return records

class DataGenerator:
    def __init__(self, records, processor, teacher_dir = TEACHER_DIR, max_answer_tokens = None):
        self.records = records
        self.processor = processor
        self.teacher_dir = teacher_dir
        # For the updating of the first K logits, our way of
        # applying KD to the student model given our memory
        # constraint an hardware problems.
        self.max_answer_tokens = max_answer_tokens
        self._cache_idx, self._cache = None, None

    def __len__(self):
        return len(self.records)

    def encode(self, idx):
        if self._cache_idx == idx:
            return self._cache
        rec = self.records[idx]
        answer = rec["answer"] if isinstance(rec["answer"], str) else json.dumps(rec["answer"], ensure_ascii = False)
        prompt = build_chat_prompt(self.processor, rec["image"])

        enc = self.processor(text = [prompt], images = [Image.open(rec["image"]).convert("RGB")], return_tensors = "np")
        prompt_ids = enc["input_ids"][0]
        answer_ids = np.array(self.processor.tokenizer(answer + END_TOKEN, add_special_tokens = False)["input_ids"])
        out = {
            "input_ids": np.concatenate([prompt_ids, answer_ids]),
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
            "answer_start": len(prompt_ids),
            "answer_ids": answer_ids,
        }
        self._cache_idx, self._cache = idx, out
        return out

    def answer_span(self, idx):
        # (first answer position, number of answer tokens used)
        # shared by the teacher script and training so the two 
        # always cover the same positions.
        enc = self.encode(idx)
        n = len(enc["answer_ids"])
        if self.max_answer_tokens is not None:
            n = min(n, self.max_answer_tokens)
        return enc["answer_start"], n

    def __getitem__(self, idx):
        enc = self.encode(idx)
        start, n = self.answer_span(idx)
        vals, ids = load_teacher_topk(self.records[idx]["teacher_key"], self.teacher_dir)
        return enc, start, enc["answer_ids"][:n], vals, ids

    def batches(self, batch_size, shuffle = True):
        idxs = list(range(len(self)))
        if shuffle:
            random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            yield self.collate([self[j] for j in idxs[i:i + batch_size]])

    @staticmethod
    def collate(samples):
        encs, starts, answers, vals, ids = zip(*samples)
        B = len(samples)
        T = max(len(e["input_ids"]) for e in encs)
        A = max(len(a) for a in answers)
        K = vals[0].shape[-1]

        # Add padding at the end
        input_ids = np.zeros((B, T), dtype = np.int32)
        labels = np.zeros((B, A), dtype = np.int32)
        mask = np.zeros((B, A), dtype = np.float32)
        t_vals = np.zeros((B, A, K), dtype = np.float32)
        t_idx = np.zeros((B, A, K), dtype = np.int32)
        for b, (e, a, v, ix) in enumerate(zip(encs, answers, vals, ids)):
            input_ids[b, :len(e["input_ids"])] = e["input_ids"]
            labels[b, :len(a)] = a
            mask[b, :len(a)] = 1.0
            t_vals[b, :len(a)] = np.array(v)[:len(a)]
            t_idx[b, :len(a)] = np.array(ix)[:len(a)]

        return {
            "inputs": {
                "input_ids": mx.array(input_ids),
                "pixel_values": mx.array(np.concatenate([e["pixel_values"] for e in encs])),
                "image_grid_thw": mx.array(np.concatenate([e["image_grid_thw"] for e in encs])),
            },
            "answer_start": mx.array(np.array(starts, dtype = np.int32)),
            "labels": mx.array(labels),
            "mask": mx.array(mask),
            "teacher_vals": mx.array(t_vals),
            "teacher_idx": mx.array(t_idx),
        }
