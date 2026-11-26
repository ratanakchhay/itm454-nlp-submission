## Playground for Knowledge Distillation

---

This folder is for the pipeline of Knowledge Distillation of the VL models chosen for the experiment inside of the project. What to expect inside of this folder:

1. **Dataset Preparation**: loading dataset, splitting and shuffling the data.
2. **Context and Prompt Engineering**: providing the context of the data, locations of the bounding boxes, and the prompt to extract the information in the images.
3. **Knowledge Distillation Pipeline**: focusing on combining the output space and the weight space transfer of knowledge to the smaller student model.
4. **Validation**: validates the performance of the student model against unseen data.

---

The architectural pipeline of the model distillation and fine-tuning:

![Diagram of the Architecture](./assets/ITM%20454%20Final%20Project.png)

---

## Folder Structure and How to Run

### Folder structure

```
playground/kd/
├── split.py              # shuffles data/images into train / test / val sub-folders
├── records.py            # builds data/record_<split>.jsonl from images + golden labels
├── prompt.py             # prompt = NID schema + OCR bbox coordinates of the image
├── ocr/                  # PaddleOCR helpers used by prompt.py to get the bboxes
├── model.py              # loads the teacher (Qwen3-VL-4B-Thinking) and student (Qwen3-VL-2B-Thinking) with mlx-vlm
├── data_generator.py     # records -> tokenised batches (image, answer, teacher top-K)
├── topk.py               # saving / loading the teacher's sparse top-K logits
├── teacher_inference.py  # teacher top-K logits (train + val) and teacher JSON (test)
├── loss.py               # hard loss (CE on gold) + soft loss (top-K KL from teacher)
├── train.py              # train_step / validation_step for the student
├── main.py               # the training loop
├── generation.py         # free generation + JSON extraction, shared by the inference scripts
├── student_inference.py  # student JSON on the test set (base or KD student)
├── eval.py               # acc / CER / WER / precision / recall / F1 against the gold JSON
├── inference.py          # running the inference of the model on a given image
├── metrics/              # eval.py output, renamed per model (metrics-teacher.json, ...) 
└── models/               # local model copies (git-ignored)

data/                     # git-ignored, contains PII
├── images/{train,test,val}/  # labelled document images, after split.py
├── golden/               # gold JSON label per image (same file name as the image)
├── unlabeled/            # images without a gold label (moved here by records.py)
├── record_{train,val,test}.jsonl  # one {"image", "answer"} per line
├── teacher/              # <image_name>.npz top-K teacher logits
├── json_output/
│   ├── test/             # gold JSON of the test images (used by eval.py)
│   ├── teacher/          # teacher predictions
│   ├── student/          # base student predictions (no KD)
│   └── kd_student/       # KD student predictions
└── output/student/       # trained student weights (student.safetensors)
```

How the pieces connect: `split.py` and `records.py` produce one record file per split. `teacher_inference.py` runs the teacher over the train and val records and writes `data/teacher/<image_name>.npz`, then generates the teacher's own JSON for the test records. `main.py` trains the student on the train records, reading the matching teacher file for each image, and keeps the epoch with the lowest val loss. The teacher is never loaded during training. `student_inference.py` then generates JSON on the test set, and `eval.py` scores any prediction folder against the gold JSON.

### Requirements

- Apple Silicon Mac (training and inference use MLX).
- Python packages: `mlx`, `mlx-vlm`, `numpy`, `pillow`, `tqdm`, `paddleocr`, `paddlepaddle`, `jiwer`, and `khmerns` (from `git+https://github.com/seanghay/khmer-neural-segmenter.git`).
- Labelled images in `data/images/` and their gold labels in `data/golden/` (same file name, `.json` extension).

### How to run

All commands are run from `playground/kd/`.

0. **Download the necessary dependencies** using requirements[mac].txt
   ```
   python3 -m venv .venv
   source .venv/bin/activate

   pip install -r requirements[mac].txt
   ```

1. **Download the models** to the paths that `model.py` expects:
   ```bash
   mkdir models # if you have not yet instantiated 

   hf download mlx-community/Qwen3-VL-2B-Thinking-bf16 --local-dir models/qwen-2b-vl-thinking-bf16
   hf download mlx-community/Qwen3-VL-4B-Thinking-bf16 --local-dir models/qwen-4b-vl-thinking-bf16
   ```

2. **Remove the unlabeled images** (only once). `records.py` moves every image with no gold label to `data/unlabeled/`. Do this before splitting so the split sizes are not changed afterwards. The `record.jsonl` it writes is not used later.
   ```bash
   python records.py -i images -g golden -o record.jsonl
   ```

3. **Split the images** (only once) into `data/images/{train,test,val}` (70 / 15 / 15 by default):
   ```bash
   python split.py -s 0
   ```

4. **Build one record file per split:**
   ```bash
   python records.py -i images/train -g golden -o record_train.jsonl
   python records.py -i images/val   -g golden -o record_val.jsonl
   python records.py -i images/test  -g golden -o record_test.jsonl
   ```

5. **Copy the gold JSON of the test images** into `data/json_output/test/`, which `eval.py` uses as ground truth:
   ```bash
   mkdir -p ../../data/json_output/test
   for f in ../../data/images/test/*; do
     n=$(basename "${f%.*}")
     cp "../../data/golden/$n.json" ../../data/json_output/test/
   done
   ```

6. **Run the teacher.** This saves the top-K logits for train + val to `data/teacher/` and the teacher's JSON for test to `data/json_output/teacher/`. The first run is slow because PaddleOCR runs on every image.
   ```bash
   python teacher_inference.py
   ```

7. **Train the student:**
   ```bash
   python main.py -e 3 -b 1 -l 1e-5
   ```
   Hard and soft loss are printed separately for train and val every epoch. The best epoch (lowest val loss) is saved to `data/output/student/student.safetensors`. Only the trainable weights are saved.

8. **Generate the student predictions on the test set**, once without KD (baseline) and once with the trained weights:
   ```bash
   python student_inference.py -o ../../data/json_output/student
   python student_inference.py -w ../../data/output/student/student.safetensors
   ```
   The second command writes to `data/json_output/kd_student/` by default.

9. **Evaluate each model.** `eval.py` always writes to `metrics/metrics.json`, so rename the file after each run:
   ```bash
   python eval.py -p ../../data/json_output/teacher    && mv metrics/metrics.json metrics/metrics-teacher.json
   python eval.py -p ../../data/json_output/student    && mv metrics/metrics.json metrics/metrics-student.json
   python eval.py -p ../../data/json_output/kd_student && mv metrics/metrics.json metrics/metrics-kd.json
   ```

Rules to keep the results valid:
- Teacher files are matched to records by image file name. Do not rename images after step 6. Reordering the record files is fine.
- If you pass `-m / --max-answer-tokens`, pass the same value to `teacher_inference.py` and `main.py`.
- If `prompt.py` or `json/nid.json` changes, rerun step 6, since the teacher and student must see the same prompt.
- Do not rerun `split.py` on an existing split. It only moves images at the top level of `data/images/`.


### Generated by Claude
