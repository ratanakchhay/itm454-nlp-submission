import os
import argparse
from tqdm import tqdm
#
import mlx.core as mx
from mlx.utils import tree_flatten
#
from model import load_student
from topk import TEACHER_DIR
from data_generator import DataGenerator, load_records, RECORD_TRAIN, RECORD_VAL
from train import build_optimizer, train_step, validation_step
#

OUT_DIR = os.path.join("..", "..", "data", "output", "student")

#

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type = int, default = 3)
parser.add_argument("-b", "--batch-size", type = int, default = 1)
parser.add_argument("-l", "--lr", type = float, default = 1e-5)
parser.add_argument("-t", "--teacher-dir", type = str, default = TEACHER_DIR)
parser.add_argument("-o", "--out", type = str, default = OUT_DIR)
parser.add_argument("-m", "--max-answer-tokens", type = int, default = None, help = "must match teacher_inference.py")
parser.add_argument("-s", "--seed", type = int, default = 0)
parser.add_argument("--train-records", type = str, default = RECORD_TRAIN)
parser.add_argument("--val-records", type = str, default = RECORD_VAL)
args = parser.parse_args()

# 

if __name__ == "__main__":
    # Instantiate the randomness, and loading the student for training
    mx.random.seed(args.seed)
    student, processor = load_student()
    # bf16 can't resolve updates this small (they round to zero), so the
    # trainable language model is kept in float32 during training
    student.model.language_model.set_dtype(mx.float32)

    # Optimizer
    optimizer = build_optimizer(args.lr)

    # Sanity Check -- hopefully I did not forget to create a folder cuz 
    # this happened like 1000000000000 times.
    os.makedirs(args.out, exist_ok = True)

    # Instantiating the batches and getting the data for training
    # The train/val split is done beforehand by split.py
    train_ds = DataGenerator(load_records(args.train_records), processor, args.teacher_dir, args.max_answer_tokens)
    val_ds = DataGenerator(load_records(args.val_records), processor, args.teacher_dir, args.max_answer_tokens)
   
    # Sanity Check for the training and validation length
    print(f"train={len(train_ds)} val={len(val_ds)}")

    # Training loop
    best = float("inf")
    for epoch in range(args.epochs):
        # Loss components are logged separately (hard = CE on gold, soft = top-K KD)
        for batch in tqdm(train_ds.batches(args.batch_size), total = -(-len(train_ds) // args.batch_size), desc = f"epoch {epoch}"):
            loss, parts = train_step(student, optimizer, batch)
        print(f"epoch {epoch} train loss={loss:.4f} hard={parts['hard_loss']:.4f} soft={parts['soft_loss']:.4f}")

        # Validation loss (mean over the val set), with hard/soft components
        val_results = [validation_step(student, b) for b in val_ds.batches(args.batch_size, shuffle = False)]
        val = sum(r[0] for r in val_results) / len(val_results)
        val_hard = sum(r[1]["hard_loss"] for r in val_results) / len(val_results)
        val_soft = sum(r[1]["soft_loss"] for r in val_results) / len(val_results)
        print(f"epoch {epoch} val loss={val:.4f} hard={val_hard:.4f} soft={val_soft:.4f}")

        # Keep the best epoch, only the trainable weights
        if val < best:
            best = val
            # Saved back in bf16, same dtype as the base model
            weights = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(student.trainable_parameters())}
            mx.save_safetensors(os.path.join(args.out, "student.safetensors"), weights)