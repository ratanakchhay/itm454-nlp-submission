import os
# 
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load

# 

# This is no longer needed as we download the model manually using hf
# os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# For the KD of Qwen family model, mostly used by research.
STUDENT_ID = os.path.join(".", "models", "qwen-2b-vl-thinking-bf16") # local copy
TEACHER_ID = os.path.join(".", "models", "qwen-4b-vl-thinking-bf16") # local copy
MAX_PIXELS = 1024 * 32 * 32
VOCAB_SIZE = 151936 # Based on HF's model information

# 

class VLMLogits(nn.Module):
    # Wraps mlx-vlm model so that model(inputs) -> logits (B, T, VOCAB_SIZE).
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, inputs):
        out = self.model(
            inputs["input_ids"],
            inputs["pixel_values"],
            image_grid_thw = inputs["image_grid_thw"],
        )
        return out.logits[..., :VOCAB_SIZE]

class _SliceHidden(nn.Module):
    # Wraps the decoder stack so only hidden states in [lo, hi) reach the LM head.
    def __init__(self, inner, lo, hi):
        super().__init__()
        self.inner, self.lo, self.hi = inner, lo, hi

    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs)[:, self.lo:self.hi]

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "inner":
                raise
            return getattr(self.inner, name)

def answer_logits(wrapped, inputs, lo, hi):
    # Logits for positions [lo, hi) only -> (hi - lo, VOCAB_SIZE). Skips the
    # (T, VOCAB_SIZE) full-sequence logits, which is what runs out of memory.
    lm = wrapped.model.language_model
    inner = lm.model
    lm.model = _SliceHidden(inner, lo, hi)
    try:
        out = wrapped.model(
            inputs["input_ids"],
            inputs["pixel_values"],
            image_grid_thw = inputs["image_grid_thw"],
        )
    finally:
        lm.model = inner
    return out.logits[0, :, :VOCAB_SIZE]

# 

def _cap_pixels(processor):
    processor.image_processor.size = {"longest_edge": MAX_PIXELS, "shortest_edge": 65536}
    return processor

def load_student(freeze_vision = True, model_id = STUDENT_ID):
    model, processor = load(model_id)
    # Checking whether to fine-tune vision aspect of the model.
    # Realistically, we are not finetuning the vision aspect
    # as we focus on its extraction.
    if freeze_vision:
        model.vision_tower.freeze()
    return VLMLogits(model), _cap_pixels(processor)

def load_teacher():
    model, processor = load(TEACHER_ID)
    model.eval()
    return VLMLogits(model), _cap_pixels(processor)
