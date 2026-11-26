import mlx.core as mx
import mlx.nn as nn
#

TEMPERATURE = 4.0
ALPHA = 0.5

#

class CustomLossFunction(nn.Module):
    def __init__(self, alpha = ALPHA, temperature = TEMPERATURE):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def __call__(self, student_logits, teacher_vals, teacher_idx, labels, mask):
        # Some aspect is checked with Claude to make sure that is correct
        # based on Qwen3's documentation.
        # 
        # student_logits: (B, A, V)  teacher_vals/teacher_idx: (B, A, K)
        # labels, mask: (B, A) -- mask is 1 on real answer tokens, 0 on padding
        mask = mask.astype(student_logits.dtype)
        n = mx.maximum(mask.sum(), 1.0)

        # Sparse top-K KD: the teacher is only known on its top-K token ids
        # so both teacher and student are renormalised over those K entries.
        teacher_vals = mx.stop_gradient(teacher_vals)
        student_topk = mx.take_along_axis(student_logits, teacher_idx, axis = -1)

        # Soft targets: KL(teacher || student) on temperature-scaled distributions
        log_p_s = student_topk / self.temperature
        log_p_s = log_p_s - mx.logsumexp(log_p_s, axis = -1, keepdims = True)
        log_p_t = teacher_vals / self.temperature
        log_p_t = log_p_t - mx.logsumexp(log_p_t, axis = -1, keepdims = True)
        #

        # kl_div_loss expects log-probabilities for both inputs and targets
        kl = nn.losses.kl_div_loss(log_p_s, log_p_t, axis = -1, reduction = "none")
        loss_soft = (kl * mask).sum() / n * (self.temperature ** 2)
       
       # Hard targets: cross entropy on the full raw student logits
        ce = nn.losses.cross_entropy(student_logits, labels, reduction = "none")
        loss_hard = (ce * mask).sum() / n
        
        #
        loss = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
        return loss, {"hard_loss": loss_hard, "soft_loss": loss_soft}
