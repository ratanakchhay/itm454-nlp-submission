import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
# 
from loss import CustomLossFunction

# 
loss_fn = CustomLossFunction()
# 

def build_optimizer(learning_rate = 1e-5, weight_decay = 0.01):
    return optim.AdamW(learning_rate = learning_rate, weight_decay = weight_decay)

def answer_logits(logits, answer_start, num_positions):
    # Extracting the answer logits directly from the output logits
    # where it would contain the prompt logits, the answer logits
    # and padding. Therefore, for comparison, we have to extract
    # the ones that are the answers.
    # 
    # Claude flagged error, code proposed by Claude.
    pos = (answer_start - 1)[:, None] + mx.arange(num_positions)[None, :]
    pos = mx.minimum(pos, logits.shape[1] - 1)
    return mx.take_along_axis(logits, pos[:, :, None], axis = 1)

def compute_loss(student, batch):
    logits = student(batch["inputs"])
    student_logits = answer_logits(logits, batch["answer_start"], batch["labels"].shape[1])
    # 
    return loss_fn(student_logits, batch["teacher_vals"], batch["teacher_idx"], batch["labels"], batch["mask"])

def train_step(student, optimizer, batch):
    student.train()

    # Calculating the loss and grad (the Apple way)
    loss_and_grad = nn.value_and_grad(student, compute_loss)
    (total_loss, total_loss_dict), grads = loss_and_grad(student, batch)

    # Clipping to get rid of gradient explosion (Gemini suggested, self-executed)
    grads, _ = optim.clip_grad_norm(grads, max_norm = 1.0)
    optimizer.update(student, grads)

    mx.eval(student.parameters(), optimizer.state, total_loss)
    return total_loss.item(), {k: v.item() for k, v in total_loss_dict.items()}

def validation_step(student, batch):
    student.eval()

    # Calculating the loss
    total_loss, total_loss_dict = compute_loss(student, batch)

    mx.eval(total_loss)
    return total_loss.item(), {k: v.item() for k, v in total_loss_dict.items()}
